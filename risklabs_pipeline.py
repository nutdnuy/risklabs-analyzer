"""RiskLabs pipeline — pure LLM feature extraction layer.

Reproduces Figures 2 (ECC Analyzer) and 3 (News Enrichment Pipeline) from:
  Cao et al. (2024-2025), "RiskLabs: Predicting Financial Risk Using Large
  Language Model based on Multimodal and Multi-Sources Data", arXiv:2404.07452

Output: qualitative risk assessment + structured extracted features.
NOT a numerical MSE-graded volatility prediction (requires trained Wav2Vec2 + BiLSTM
weights unavailable outside the original paper's environment).
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class ECCAnalysis(BaseModel):
    """Structured output for the ECC Analyzer (paper Fig. 2)."""

    overview_summary: str = Field(
        description="2-3 sentence high-level narrative of the earnings call"
    )
    performance_summary: str = Field(
        description="Key financial metrics and performance discussed (revenue, EPS, margins, guidance)"
    )
    topic_summary: str = Field(
        description="Other key themes: macro environment, strategy, ops, competition, etc."
    )

    # Question Bank
    cash_flow_drivers: str = Field(
        description="What are the primary drivers of cash flow according to management?"
    )
    rising_cost_impact: str = Field(
        description="How do rising costs (labor, materials, rates) impact the business model?"
    )
    dividend_outlook: str = Field(
        description="Dividend/capital return scenario: one of Increasing/Decreasing/Stable + reasoning"
    )

    # Expert analysis
    innovation_analysis: str = Field(
        description="New products, R&D investments, patents, or technology launches mentioned"
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="0-5 red flags: fraud signals, impairments, guidance cuts, restructuring, "
                    "missed targets, auditor concerns, regulatory scrutiny, evasive answers",
    )
    confidence_signals: list[str] = Field(
        default_factory=list,
        description="0-5 positive signals: raised guidance, buybacks, new products, "
                    "market share gains, strong pipeline, management conviction",
    )


class NewsSignal(BaseModel):
    """Structured output for a single news item (paper Fig. 3)."""

    headline: str = Field(description="Short headline or first sentence of the news item")
    sentiment: str = Field(
        description="Overall sentiment: one of positive, negative, neutral"
    )
    financial_performance: str = Field(
        description="Financial performance signal: one of increasing, decreasing, stable, NaN"
    )
    financial_topic: str = Field(
        description="Primary financial topic: one of cash_flow, earnings, stock_price, NaN"
    )
    regulatory_issue: str = Field(
        description="Regulatory/legal issue: one of fraud, lawsuit, antitrust, quality_concerns, none, NaN"
    )
    innovation_signal: str = Field(
        description="Innovation signal: one of new_product, new_patent, R&D, none, NaN"
    )
    market_response_prediction: str = Field(
        description="1-sentence predicted near-term market response to this news"
    )


class NewsEnrichment(BaseModel):
    """Batch news enrichment output (paper Fig. 3)."""

    signals: list[NewsSignal] = Field(
        description="One NewsSignal per input news item, in the same order"
    )


class RiskAssessment(BaseModel):
    """Final risk synthesis combining ECC + News signals."""

    risk_level: str = Field(
        description="Overall risk level: one of HIGH, MEDIUM, LOW"
    )
    risk_score: int = Field(
        description="Risk score 0-100 where higher = more risk. 0-35=low, 36-65=medium, 66-100=high",
        ge=0,
        le=100,
    )
    confidence: str = Field(
        description="Confidence in this assessment: one of low, medium, high"
    )
    primary_risk_factors: list[str] = Field(
        description="3-6 bullet items describing the main risk factors identified"
    )
    mitigating_factors: list[str] = Field(
        default_factory=list,
        description="0-5 bullet items describing factors that reduce risk"
    )
    volatility_outlook: dict[str, str] = Field(
        description="Volatility outlook per horizon. Keys: 3d, 7d, 15d, 30d. "
                    "Values: 'elevated — <short reason>' / 'moderate — <short reason>' / 'subdued — <short reason>'"
    )
    narrative: str = Field(
        description="3-4 sentence overall risk narrative integrating ECC signals and news"
    )


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

ECC_SYSTEM_PROMPT = """You are an expert financial analyst specializing in earnings call transcript analysis for institutional risk management.

Your task is to analyze earnings call transcripts using the hierarchical summarization strategy described in academic literature on LLM-based financial risk analysis (Cao et al. 2024):

STEP 1 — CHUNKED SUMMARIZATION:
Split the transcript conceptually into three sections:
- Overview: company-wide narrative, strategic messaging, tone
- Performance: specific financial metrics, KPIs, guidance numbers
- Topics: operational themes, competitive landscape, macro factors, Q&A themes

STEP 2 — QUESTION BANK ANALYSIS:
Answer these risk-focused questions from the transcript:
Q1: What are the primary drivers of cash flow according to management?
Q2: How do rising costs (labor, materials, interest rates) impact the business?
Q3: What is the dividend/capital return scenario? (Classify as Increasing/Decreasing/Stable)

STEP 3 — EXPERT RISK FLAG DETECTION:
Identify risk flags (0-5 items). Flag these patterns if present:
- Guidance cuts or below-consensus outlook
- Restructuring charges or impairments
- Missed earnings/revenue targets
- Regulatory scrutiny or legal proceedings
- Auditor concerns or accounting issues
- Evasive non-answers to analyst questions
- Abnormal management turnover mentioned
- Fraud language or internal investigation hints

STEP 4 — CONFIDENCE SIGNAL DETECTION:
Identify confidence signals (0-5 items). Flag these positives if present:
- Raised full-year guidance
- Share buyback announcements
- New product launches with concrete timelines
- Market share gains
- Strong order backlog or pipeline
- High management conviction with specific data

CRITICAL INSTRUCTION: Preserve contextual qualifiers in all extracted items. Write "North America cloud revenue growth" not "revenue". Write "gross margin pressure from semiconductor supply constraints" not "margin pressure". Specificity is the paper's core innovation over generic NER.

Return a fully structured ECCAnalysis object."""

NEWS_SYSTEM_PROMPT = """You are an expert financial news analyst applying the four-step enrichment pipeline from RiskLabs (Cao et al. 2024, arXiv:2404.07452, Fig. 3).

For EACH news item provided, extract structured signals using EXACTLY these enum values:

STEP 1 — SENTIMENT:
Classify: positive | negative | neutral
(Consider the company's financial prospects, not headline drama)

STEP 2 — FINANCIAL PERFORMANCE SIGNAL:
Classify: increasing | decreasing | stable | NaN
(Does this news imply improving, deteriorating, or flat financial performance? NaN if not financial.)

STEP 3 — FINANCIAL TOPIC:
Classify: cash_flow | earnings | stock_price | NaN
(Primary topic. NaN if this news is not directly about a financial metric.)

STEP 4 — REGULATORY ISSUE:
Classify: fraud | lawsuit | antitrust | quality_concerns | none | NaN
(fraud: accounting/securities fraud; antitrust: competition/monopoly; quality_concerns: product recalls/defects; none: no issue; NaN: unclear)

STEP 5 — INNOVATION SIGNAL:
Classify: new_product | new_patent | R&D | none | NaN
(new_product: launch announced; new_patent: patent filing/grant; R&D: investment/collaboration; none: no innovation; NaN: unclear)

STEP 6 — MARKET RESPONSE PREDICTION:
Write exactly 1 sentence predicting the near-term (1-5 day) market response to this specific news item.

CRITICAL: Return EXACTLY one NewsSignal per input news item, in the same order. Never skip items.
Use "NaN" (the string) for categories that don't apply or cannot be determined from the news text.
Preserve the exact enum spellings above."""

SYNTHESIS_SYSTEM_PROMPT = """You are a senior financial risk officer synthesizing multiple information sources into a unified risk assessment.

You receive:
1. ECC Analysis — structured analysis of an earnings call transcript (strongest weight)
2. News Enrichment — structured signals from recent news items
3. Optional market context provided by the user

WEIGHTING HIERARCHY (from most to least impactful):
1. HIGHEST: Risk flags from earnings call (each flag = strong negative signal)
2. HIGH: Regulatory issues in news (fraud > antitrust/lawsuit > quality_concerns)
3. MEDIUM: Negative sentiment cluster in news (3+ negative signals = systematic concern)
4. MEDIUM: Decreasing financial performance signals in news
5. LOW: Positive signals from earnings (can mitigate but rarely offset red flags)
6. LOW: Innovation signals (positive but long-term)

RISK SCORE CALIBRATION:
- 0-35 (LOW): No risk flags, positive guidance, clean news
- 36-65 (MEDIUM): 1-2 risk flags OR regulatory news without earnings confirmation
- 66-85 (HIGH): 2+ risk flags OR fraud/regulatory in both earnings + news
- 86-100 (VERY HIGH): Multiple severe risk flags + confirmed regulatory issues

VOLATILITY OUTLOOK HORIZONS (3d, 7d, 15d, 30d):
- Regulatory/fraud: elevated at 3d, 7d; moderate at 15d; subdued at 30d (uncertainty peak is immediate)
- Guidance cut: elevated at 3d; moderate at 7d, 15d; subdued at 30d (reprices quickly)
- Mixed signals: moderate across all horizons
- Clean signals: subdued across all horizons, moderate at 30d for macro drift

Format volatility values as: "elevated — <10-word reason>" / "moderate — <10-word reason>" / "subdued — <10-word reason>"

Return a complete RiskAssessment object. Primary risk factors and mitigating factors should be specific, not generic."""

# ---------------------------------------------------------------------------
# Demo Scenarios
# ---------------------------------------------------------------------------

DEMO_SCENARIOS: dict[str, dict] = {
    "🔴 High risk — missed guidance + regulatory": {
        "description": "Tech company misses estimates, faces SEC inquiry, key executive departs",
        "transcript": """Good morning, and thank you for joining NovaTech's Q3 2024 earnings call. I'm David Chen, CFO.

Before we begin, I want to address the elephant in the room. Q3 results came in below our prior guidance range. Revenue was $2.1 billion versus guidance of $2.3-2.4 billion, and adjusted EPS of $0.87 missed the Street consensus of $1.12 by a significant margin.

The shortfall was driven by several factors. First, our enterprise software segment saw unexpected elongation of sales cycles in North America. Deals we expected to close in Q3 have pushed into Q4 and even Q1 of next year. Second, our cloud infrastructure margins contracted sharply due to elevated compute costs from our GenAI initiative, which has not yet reached the efficiency thresholds we had projected.

On the personnel side, I want to inform investors that our Chief Revenue Officer, Sarah Martinez, has decided to pursue other opportunities. We wish her well. The search for her replacement is underway.

Regarding the previously disclosed informal inquiry from the SEC regarding our revenue recognition practices in the federal contracts division — we are cooperating fully and have engaged outside counsel. We believe our practices are fully compliant, but we cannot predict the timeline or outcome.

Looking at Q4 guidance, we are withdrawing our prior full-year outlook and will not provide new guidance until we have better visibility on the enterprise pipeline. This is a difficult decision but we believe transparency is paramount.

In terms of restructuring, the Board has approved a workforce reduction of approximately 8% to reduce our cost base. We expect to take a charge of approximately $180-220 million in Q4.

Analyst questions:

Q: Can you be more specific about which geographies are seeing softness?
A: We're seeing it broadly, frankly. I'd rather not get into specific regional breakdowns right now.

Q: Has the SEC inquiry expanded beyond the federal contracts division?
A: We can't comment on the specifics while it's ongoing. We're cooperating.

Q: What gives you confidence in the business going into next year given these challenges?
A: We have a strong product roadmap and believe the market opportunity is intact. The near-term headwinds are real but we're taking the right actions.

Thank you for your questions. We appreciate your continued support.""",
        "news_items": [
            "NovaTech Corp (NVTC) shares plunged 18% in pre-market trading following the company's Q3 earnings miss and guidance withdrawal. The company reported revenue of $2.1 billion versus the $2.3 billion consensus estimate, and adjusted EPS of $0.87 missed the $1.12 consensus by 22%. CFO David Chen declined to provide forward guidance, citing visibility concerns. The stock has now fallen 34% year-to-date.",
            "The U.S. Securities and Exchange Commission has expanded its informal inquiry into NovaTech Corp's revenue recognition practices to include its commercial contracts division, according to two people familiar with the matter. The SEC is examining whether NovaTech improperly accelerated recognition of multi-year software license agreements. The company previously disclosed only an inquiry limited to federal contracts. NovaTech shares fell an additional 6% on the news.",
            "NovaTech Chief Revenue Officer Sarah Martinez has departed the company, the company confirmed Thursday. Martinez, who joined NovaTech three years ago from Salesforce, led the enterprise sales organization responsible for the segment that missed Q3 targets. The departure follows similar exits of the Chief Marketing Officer in August and the President of EMEA in June. Three C-suite departures in five months raises governance concerns, analysts said.",
            "CloudScale Partners, NovaTech's largest direct competitor in enterprise workflow software, reported Q3 results that beat consensus by 15% and raised full-year guidance by $300 million. CEO Amy Johnson cited strong enterprise demand and improving GenAI monetization. 'We are taking market share from incumbents who are struggling to execute,' Johnson said on the earnings call, without naming NovaTech directly. Analysts noted that CloudScale's win rate against NovaTech has reportedly increased from 40% to 58% over the past year."
        ],
        "context": "VIX at 24, tech sector under pressure from rising rates and AI monetization uncertainty"
    },
    "🟢 Low risk — strong quarter + innovation": {
        "description": "Consumer goods company beats estimates, raises guidance, launches new product",
        "transcript": """Good morning, everyone. Thank you for joining BrightBrand's Q3 2024 earnings call. I'm Jennifer Walsh, CEO, joined by our CFO, Michael Torres.

I'm delighted to share that Q3 was an exceptional quarter across every dimension. Revenue came in at $4.8 billion, above our guidance range of $4.4-4.6 billion, representing 14% organic growth year-over-year. This was our fifth consecutive quarter of double-digit organic growth.

Adjusted EBITDA was $1.1 billion, yielding a margin of 22.9%, which is a 180 basis point expansion versus Q3 last year. Adjusted EPS of $2.34 beat consensus of $2.05 by 14%.

North America was our standout performer with 18% organic growth, driven by our premium home care line and our new Health Plus segment. International grew 10% with particular strength in Southeast Asia, where our distribution partnership with MegaMart delivered strong early results.

On capital allocation, our Board has approved a 15% increase in the quarterly dividend, bringing it to $0.69 per share. We're also announcing a new $2 billion share repurchase authorization. These reflect our confidence in the cash flow durability of our business.

Looking at innovation, I'm thrilled to announce BrightHome Pro — our AI-powered smart home ecosystem launching in Q1 2025. We've filed 23 patents covering the core technology stack. Early beta testing with 50,000 households showed 91% satisfaction and an NPS of 74, which is exceptional for a new product category.

Full-year guidance is being raised. We now expect organic revenue growth of 12-13% versus our prior outlook of 10-12%. Adjusted EPS guidance is raised to $8.90-9.10 from $8.40-8.70.

Analyst Q&A:

Q: Can you quantify the Health Plus revenue?
A: Health Plus contributed $380 million in Q3, up from $180 million in Q3 last year. We're on track for $1.2 billion full-year, ahead of our $1 billion target.

Q: What's the incremental margin profile of BrightHome Pro?
A: We expect gross margins in the 60-65% range, well above our company average. This will be accretive to overall margins by 2026.

Q: Are you seeing any supply chain pressure?
A: Our supply chain is operating extremely well. We secured multi-year contracts with key suppliers last year that give us both cost certainty and volume priority.

Thank you all. It was a terrific quarter.""",
        "news_items": [
            "Morgan Stanley upgraded BrightBrand (BRBT) to Overweight from Equal Weight, raising the price target to $185 from $145. Analyst Claire Fontaine cited the Q3 beat and raised guidance as evidence that BrightBrand's transformation into a premium consumer technology company is ahead of schedule. 'The Health Plus segment is scaling faster than we modeled, and BrightHome Pro could be a $3 billion revenue opportunity by 2028,' Fontaine wrote.",
            "BrightBrand announced a strategic partnership with Samsung to integrate its BrightHome Pro platform into Samsung's SmartThings ecosystem, giving the product access to Samsung's 350 million active device users. The partnership includes Samsung taking a 5% equity stake in BrightBrand's Digital Products division for $450 million, implying a $9 billion valuation for that unit alone.",
            "BrightBrand's Q3 free cash flow of $820 million exceeded consensus by 23%, driven by working capital efficiency and strong North America margins. The company ended the quarter with $3.2 billion in net cash, zero commercial paper borrowings, and its credit facility undrawn. S&P revised BrightBrand's credit outlook to Positive from Stable, a potential precursor to a full upgrade to AA.",
            "BrightBrand's new Health Plus supplement line has received FDA Generally Recognized as Safe (GRAS) certification for its three flagship products, clearing a key regulatory milestone for accelerated retail distribution. Walmart and Target have both confirmed planogram allocations of 12 SKUs each beginning January 2025, the companies' largest initial shelf space allocation for a new consumer health entrant in five years."
        ],
        "context": "Stable macro environment, consumer spending resilient, VIX at 14"
    },
    "🟡 Mixed — solid quarter, emerging concerns": {
        "description": "Cloud software beats earnings but flags slowing growth; news includes class action and CFO change",
        "transcript": """Thank you, everyone. I'm Robert Kim, CEO of DataStream Inc. Joining me is our new CFO, Patricia Lee, who joined us six weeks ago from FinServ Capital.

Q3 results were solid. Total revenue grew 22% year-over-year to $1.48 billion, ahead of our guidance midpoint of $1.44 billion. Non-GAAP operating income was $312 million, or a 21% margin, which is within our target range.

I do want to provide some context on our growth trajectory. Our core data analytics platform grew 22% as reported, but I want to be transparent that growth in our cloud-native segment — which is the future of the business — decelerated from 38% in Q2 to 29% in Q3. We believe this is a normalization from a period of elevated demand post-COVID, but we are watching it carefully.

Net Revenue Retention was 114%, down from 122% in Q2. This is within a healthy range, but the trend is something we're focused on. Some customers have reduced seat counts as part of their own cost optimization initiatives.

Our new AI Analytics module, which we launched in August, has been added by 680 customers in its first full quarter. Average contract value for AI Analytics customers is 40% higher than our standard platform, which we believe will improve net retention going forward.

On cash flow: operating cash flow was $280 million and free cash flow was $195 million after higher-than-expected capex. We raised capex guidance by $150 million for the year to fund our new AI infrastructure build-out.

Q4 guidance: we expect revenue of $1.52-1.56 billion and non-GAAP EPS of $0.84-0.89. This reflects a continuation of current trends.

Patricia, our new CFO, will be available for investor meetings starting next month as she completes her onboarding.

Q&A:

Q: Can you reconcile the NRR decline? Is it concentrated in any customer tier?
A: We saw the most pressure in the 50-200 seat range. Enterprise customers 1,000+ seats remained very stable. Mid-market is where customers are being most cautious on expansion.

Q: The AI module metrics are encouraging, but 680 out of your 8,400 customer base is only 8%. Why aren't more customers adopting?
A: Adoption takes time. We're focused on the quality of adoption, not quantity. Customers who've adopted AI Analytics have seen strong ROI which leads to organic word of mouth.

Q: Patricia, any initial impressions from your first six weeks?
A: I'm still in learning mode, honestly. The business fundamentals are strong. I'll have more to say after Q4 earnings.

Thank you for your continued interest in DataStream.""",
        "news_items": [
            "A securities class action lawsuit was filed against DataStream Inc. (DSTR) in the Southern District of New York, alleging the company misled investors about the sustainability of its cloud-native growth trajectory. The complaint, filed on behalf of shareholders who purchased stock between January and September 2024, claims company officials made materially false statements about customer demand while insiders sold approximately $85 million in stock. DataStream said in a statement it believes the lawsuit is without merit and intends to defend vigorously.",
            "DataStream's new CFO Patricia Lee disclosed in an SEC Form 4 filing that she purchased 50,000 shares of DataStream stock at $68.40 per share on her first day at the company, for a total investment of approximately $3.42 million. The purchase, funded from her personal assets, signals confidence in the company's prospects. DataStream shares closed at $71.15 on the day of the filing.",
            "Gartner named DataStream Inc. a Leader in its 2024 Magic Quadrant for Cloud Data Analytics Platforms, placing it in the highest position for Completeness of Vision for the third consecutive year. The company's AI Analytics module received special recognition for its natural language query interface. Gartner cited 'unprecedented ease of use for non-technical users' as a key differentiator.",
            "Industry tracker CloudMetrics reported that DataStream's market share in the mid-market cloud analytics segment fell from 28% to 24% over the past 12 months, as competitors Snowflake and Databricks expanded their product-led growth motions. However, DataStream maintained its 41% share in large enterprise, where switching costs are high and long-term contracts are the norm."
        ],
        "context": "SaaS multiples compressed, investors focused on profitable growth and NRR trends"
    },
}

# ---------------------------------------------------------------------------
# Pipeline Functions
# ---------------------------------------------------------------------------


def analyze_ecc(transcript: str, llm: BaseChatModel) -> ECCAnalysis:
    """Analyze an earnings call transcript (paper Fig. 2 — ECC Analyzer).

    Uses hierarchical summarization strategy + Question Bank + expert risk detection.
    Returns structured ECCAnalysis via LLM structured output.
    """
    structured_llm = llm.with_structured_output(ECCAnalysis)
    messages = [
        {"role": "system", "content": ECC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please analyze the following earnings call transcript:\n\n---\n{transcript}\n---",
        },
    ]
    return structured_llm.invoke(messages)


def enrich_news(news_items: list[str], llm: BaseChatModel) -> NewsEnrichment:
    """Apply the four-step news enrichment pipeline (paper Fig. 3).

    Batches ALL news items in ONE LLM call to minimize cost.
    Returns structured NewsEnrichment with one NewsSignal per input item.
    """
    if not news_items:
        return NewsEnrichment(signals=[])

    items_formatted = "\n\n".join(
        f"[NEWS ITEM {i+1}]\n{item.strip()}" for i, item in enumerate(news_items)
    )
    structured_llm = llm.with_structured_output(NewsEnrichment)
    messages = [
        {"role": "system", "content": NEWS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Please enrich the following {len(news_items)} news items. "
                f"Return EXACTLY {len(news_items)} NewsSignal objects in the signals list.\n\n"
                f"{items_formatted}"
            ),
        },
    ]
    return structured_llm.invoke(messages)


def synthesize_risk(
    ecc: ECCAnalysis,
    news: NewsEnrichment,
    context: str,
    llm: BaseChatModel,
) -> RiskAssessment:
    """Synthesize ECC analysis + news signals into a final risk assessment."""
    # Build a concise summary of inputs for the synthesis prompt
    flags_str = "\n".join(f"  - {f}" for f in ecc.risk_flags) if ecc.risk_flags else "  (none)"
    signals_str = "\n".join(f"  - {s}" for s in ecc.confidence_signals) if ecc.confidence_signals else "  (none)"

    news_summary_lines = []
    for i, sig in enumerate(news.signals):
        news_summary_lines.append(
            f"  News {i+1}: [{sig.sentiment.upper()}] | financial_perf={sig.financial_performance} | "
            f"regulatory={sig.regulatory_issue} | innovation={sig.innovation_signal}"
        )
    news_summary = "\n".join(news_summary_lines) if news_summary_lines else "  (no news signals)"

    user_content = f"""ECC ANALYSIS SUMMARY:
Overview: {ecc.overview_summary}
Performance: {ecc.performance_summary}
Topics: {ecc.topic_summary}
Cash flow drivers: {ecc.cash_flow_drivers}
Rising cost impact: {ecc.rising_cost_impact}
Dividend outlook: {ecc.dividend_outlook}
Innovation: {ecc.innovation_analysis}

RISK FLAGS (ECC):
{flags_str}

CONFIDENCE SIGNALS (ECC):
{signals_str}

NEWS SIGNALS:
{news_summary}

MARKET CONTEXT: {context if context.strip() else "Not provided"}

Please synthesize the above into a comprehensive RiskAssessment."""

    structured_llm = llm.with_structured_output(RiskAssessment)
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return structured_llm.invoke(messages)


def interpret_risk_level(score: int) -> tuple[str, str]:
    """Map 0-100 risk score to (label with emoji, narrative description).

    Based on paper's Q1-Q5 alpha quintile findings translated to qualitative risk bands.
    """
    if score <= 25:
        label = "LOW RISK"
        narrative = (
            "Strong financial position with limited downside catalysts. "
            "This profile aligns with low-volatility names that tend to exhibit "
            "subdued realized volatility and positive risk-adjusted returns. "
            "Suitable for lower-risk allocation with standard monitoring cadence."
        )
    elif score <= 45:
        label = "MODERATE-LOW RISK"
        narrative = (
            "Generally healthy fundamentals with minor concerns noted. "
            "Near-term volatility is expected to be contained. "
            "Monitor for emerging developments but no immediate action threshold reached."
        )
    elif score <= 65:
        label = "MEDIUM RISK"
        narrative = (
            "Mixed signals present — positives partially offset by identifiable risks. "
            "Elevated attention warranted. Consider position sizing adjustments and "
            "increased monitoring frequency. Near-term volatility may be above average."
        )
    elif score <= 80:
        label = "HIGH RISK"
        narrative = (
            "Multiple risk factors detected across earnings and/or news channels. "
            "This profile is associated with elevated realized volatility and "
            "potential left-tail outcomes. Risk mitigation strategies recommended. "
            "Review position limits and hedging exposure."
        )
    else:
        label = "VERY HIGH RISK"
        narrative = (
            "Severe risk signals present — governance concerns, regulatory exposure, "
            "or fundamental deterioration detected. This profile is consistent with "
            "maximum volatility and potential systemic stress. "
            "Immediate risk escalation to portfolio management recommended."
        )
    return label, narrative
