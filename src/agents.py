"""
agents.py
=========
All 5 Agent implementations for the Agentic AI Healthcare System.
Author : Sirigiri Venkateswara Adithya | IIIT Sonepat
Faculty: Dr. K Naveen Kumar
"""

from __future__ import annotations
import json, os, textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

try:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class PatientRecord:
    patient_id: str
    age: int
    sex: str
    bmi: float
    spo2_mean: float
    spo2_min: float
    nasal_airflow_mean: float
    thoracic_effort_mean: float
    ahi: Optional[float] = None
    notes: str = ""

@dataclass
class AgentOutput:
    agent_name: str
    status: str                        # "success" | "warning" | "error"
    data: Dict[str, Any]               = field(default_factory=dict)
    reasoning: str                     = ""
    recommendations: List[str]         = field(default_factory=list)


# ── 1. DATA AGENT ─────────────────────────────────────────────────────────────

class DataAgent:
    REQUIRED_FIELDS = [
        "patient_id", "age", "sex", "bmi",
        "spo2_mean", "spo2_min",
        "nasal_airflow_mean", "thoracic_effort_mean",
    ]
    VALID_RANGES = {
        "age":                   (0,   120),
        "bmi":                   (10,  80),
        "spo2_mean":             (70,  100),
        "spo2_min":              (50,  100),
        "nasal_airflow_mean":    (0,   1),
        "thoracic_effort_mean":  (0,   1),
    }

    def process(self, raw: Dict[str, Any]) -> AgentOutput:
        # 1. Field presence check
        missing = [f for f in self.REQUIRED_FIELDS if f not in raw]
        if missing:
            return AgentOutput(
                agent_name="DataAgent", status="error",
                reasoning=f"Missing required fields: {missing}"
            )

        # 2. Range validation
        warnings = []
        for field_name, (lo, hi) in self.VALID_RANGES.items():
            val = float(raw.get(field_name, 0))
            if not (lo <= val <= hi):
                warnings.append(
                    f"Field '{field_name}' = {val} is outside expected range [{lo}, {hi}]."
                )

        # 3. Feature engineering
        spo2_drop = raw["spo2_mean"] - raw["spo2_min"]
        odi_proxy = spo2_drop / 3.0
        fli       = (1.0 - raw["nasal_airflow_mean"]) * raw["thoracic_effort_mean"]

        record = PatientRecord(
            patient_id            = str(raw["patient_id"]),
            age                   = int(raw["age"]),
            sex                   = str(raw["sex"]).upper(),
            bmi                   = float(raw["bmi"]),
            spo2_mean             = float(raw["spo2_mean"]),
            spo2_min              = float(raw["spo2_min"]),
            nasal_airflow_mean    = float(raw["nasal_airflow_mean"]),
            thoracic_effort_mean  = float(raw["thoracic_effort_mean"]),
            ahi                   = raw.get("ahi"),
            notes                 = str(raw.get("notes", "")),
        )

        return AgentOutput(
            agent_name = "DataAgent",
            status     = "warning" if warnings else "success",
            data = {
                "record": asdict(record),
                "derived": {
                    "spo2_drop_pct":         round(spo2_drop, 2),
                    "odi_proxy_per_hr":      round(odi_proxy, 2),
                    "flow_limitation_index": round(fli, 3),
                },
            },
            reasoning = "\n".join(warnings) if warnings else "All fields validated successfully.",
        )


# ── 2. ANALYSIS AGENT ─────────────────────────────────────────────────────────

class AnalysisAgent:
    AHI_THRESHOLDS = {
        "None":     (0,  5),
        "Mild":     (5,  15),
        "Moderate": (15, 30),
        "Severe":   (30, float("inf")),
    }

    def _estimate_ahi(self, derived, record) -> float:
        odi   = derived["odi_proxy_per_hr"]
        fli   = derived["flow_limitation_index"]
        bmi_f = max(0.0, (record["bmi"] - 25) / 10)
        return round(odi * 3.5 + fli * 20 + bmi_f * 2, 1)

    def _classify_severity(self, ahi: float) -> str:
        for label, (lo, hi) in self.AHI_THRESHOLDS.items():
            if lo <= ahi < hi:
                return label
        return "Unknown"

    def process(self, data_output: AgentOutput) -> AgentOutput:
        if data_output.status == "error":
            return AgentOutput(
                agent_name="AnalysisAgent", status="error",
                reasoning="Upstream DataAgent reported an error."
            )

        record  = data_output.data["record"]
        derived = data_output.data["derived"]

        ahi_known = record.get("ahi") is not None
        ahi       = record["ahi"] if ahi_known else self._estimate_ahi(derived, record)
        severity  = self._classify_severity(ahi)

        hypo_risk = (
            "High"     if record["spo2_min"] < 85 else
            "Moderate" if record["spo2_min"] < 90 else
            "Low"
        )

        fli = derived["flow_limitation_index"]
        apnea_type = (
            "Obstructive (OSA)"    if fli > 0.35 else
            "Central (CSA) proxy"  if fli < 0.15 and derived["odi_proxy_per_hr"] > 5 else
            "Mixed / Undetermined"
        )

        recs = []
        if severity in ("Moderate", "Severe"):
            recs.append("Urgent referral to sleep specialist recommended.")
        if hypo_risk == "High":
            recs.append("Supplemental oxygen assessment warranted.")
        if "Obstructive" in apnea_type and ahi >= 15:
            recs.append("CPAP therapy evaluation advised.")

        return AgentOutput(
            agent_name = "AnalysisAgent",
            status     = "success",
            data = {
                "ahi":             ahi,
                "ahi_source":      "measured" if ahi_known else "estimated",
                "severity":        severity,
                "hypoxaemia_risk": hypo_risk,
                "apnea_type":      apnea_type,
            },
            reasoning = (
                f"AHI: {ahi:.1f} events/hr ({'measured' if ahi_known else 'estimated'})\n"
                f"Severity: {severity} | Hypoxaemia: {hypo_risk} | Apnea: {apnea_type}\n"
                f"FLI: {fli:.3f} | ODI proxy: {derived['odi_proxy_per_hr']:.1f}"
            ),
            recommendations = recs,
        )


# ── 3. KNOWLEDGE RETRIEVAL AGENT ─────────────────────────────────────────────

class KnowledgeRetrievalAgent:
    KNOWLEDGE_BASE = {
        "severity_guidelines": {
            "None":     "No treatment required; lifestyle counselling for high-risk factors.",
            "Mild":     "Conservative management: weight loss, positional therapy, oral appliances.",
            "Moderate": "CPAP therapy first line; oral appliance for CPAP-intolerant patients.",
            "Severe":   "Immediate CPAP/BiPAP initiation; consider surgical evaluation (UPPP/MMA).",
        },
        "comorbidity_risks": {
            "hypertension":   "OSA doubles hypertension risk; CPAP reduces nocturnal BP by 2-3 mmHg.",
            "cardiovascular": "Moderate-severe OSA associated with 2x increased AF risk.",
            "diabetes_t2":    "Untreated OSA worsens insulin resistance; AHI correlates with HbA1c.",
            "stroke":         "Severe OSA (AHI > 30) increases stroke risk by ~3x (SHHS data).",
        },
        "cpap_efficacy": "CPAP reduces AHI by >50% in >80% of patients (Giles et al. 2006).",
        "ahi_definitions": {
            "apnea":    "Complete cessation of airflow >= 10 seconds.",
            "hypopnea": ">=30% reduction in airflow with >=3% SpO2 drop or arousal (AASM 2012).",
            "ahi":      "Number of apnea + hypopnea events per hour of sleep.",
        },
    }

    def process(self, analysis_output: AgentOutput) -> AgentOutput:
        if analysis_output.status == "error":
            return AgentOutput(
                agent_name="KnowledgeRetrievalAgent", status="error",
                reasoning="Cannot retrieve knowledge: upstream error."
            )

        severity = analysis_output.data.get("severity", "Unknown")
        ahi      = analysis_output.data.get("ahi", 0)

        guideline     = self.KNOWLEDGE_BASE["severity_guidelines"].get(severity, "Consult specialist.")
        comorbidities = []
        if ahi >= 15:
            comorbidities.append(self.KNOWLEDGE_BASE["comorbidity_risks"]["hypertension"])
            comorbidities.append(self.KNOWLEDGE_BASE["comorbidity_risks"]["cardiovascular"])
        if ahi >= 30:
            comorbidities.append(self.KNOWLEDGE_BASE["comorbidity_risks"]["stroke"])

        cpap_note = self.KNOWLEDGE_BASE["cpap_efficacy"] if ahi >= 15 else ""

        return AgentOutput(
            agent_name = "KnowledgeRetrievalAgent",
            status     = "success",
            data = {
                "severity_guideline": guideline,
                "comorbidity_risks":  comorbidities,
                "cpap_efficacy_note": cpap_note,
                "ahi_definitions":    self.KNOWLEDGE_BASE["ahi_definitions"],
            },
            reasoning = (
                f"Retrieved guideline for '{severity}' severity. "
                f"Identified {len(comorbidities)} comorbidity risk(s)."
            ),
        )


# ── 4. RECOMMENDATION AGENT ───────────────────────────────────────────────────

class RecommendationAgent:
    def __init__(self):
        self._llm = None
        if _LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self._llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
            except Exception:
                self._llm = None

    def _heuristic(self, record, analysis, knowledge) -> str:
        sev   = analysis.data.get("severity", "Unknown")
        ahi   = analysis.data.get("ahi", 0)
        hypo  = analysis.data.get("hypoxaemia_risk", "Unknown")
        atype = analysis.data.get("apnea_type", "Unknown")
        guide = knowledge.data.get("severity_guideline", "")
        comor = knowledge.data.get("comorbidity_risks", [])
        cpap  = knowledge.data.get("cpap_efficacy_note", "")

        lines = [
            f"Patient {record['patient_id']} ({record['sex']}, {record['age']} yrs, "
            f"BMI {record['bmi']:.1f}) presents with {sev} Sleep Apnea "
            f"(AHI ~ {ahi:.1f} events/hr, {analysis.data.get('ahi_source','')}).",
            "",
            f"Apnea type: {atype}.",
            f"Nocturnal hypoxaemia risk: {hypo} (min SpO2 = {record['spo2_min']}%).",
            "",
            f"Clinical guideline: {guide}",
        ]
        if cpap:
            lines += ["", f"Evidence note: {cpap}"]
        if comor:
            lines += ["", "Comorbidity risks:"]
            lines += [f"  - {c}" for c in comor]
        return "\n".join(lines)

    def _llm_summary(self, record, analysis, knowledge) -> str:
        context = json.dumps({
            "patient":   record,
            "analysis":  analysis.data,
            "knowledge": knowledge.data,
        }, indent=2)
        try:
            response = self._llm([
                SystemMessage(content=(
                    "You are a clinical AI assistant specialising in sleep medicine. "
                    "Given structured patient data and analysis, produce a concise "
                    "3-5 sentence clinical summary followed by numbered management "
                    "recommendations. Use plain medical English. Do not hallucinate."
                )),
                HumanMessage(content=f"Patient context:\n{context}\n\nGenerate the clinical summary.")
            ])
            return response.content
        except Exception:
            return self._heuristic(record, analysis, knowledge)

    def process(self, data_output, analysis_output, knowledge_output) -> AgentOutput:
        if any(o.status == "error" for o in [data_output, analysis_output, knowledge_output]):
            return AgentOutput(
                agent_name="RecommendationAgent", status="error",
                reasoning="One or more upstream agents reported an error."
            )

        record = data_output.data["record"]
        if self._llm:
            summary = self._llm_summary(record, analysis_output, knowledge_output)
            method  = "LLM-powered (GPT-3.5-turbo via LangChain)"
        else:
            summary = self._heuristic(record, analysis_output, knowledge_output)
            method  = "Heuristic (rule-based; LLM unavailable)"

        # Deduplicate recommendations from all upstream agents
        all_recs    = analysis_output.recommendations + knowledge_output.recommendations
        seen        = set()
        unique_recs = []
        for r in all_recs:
            if r not in seen:
                seen.add(r)
                unique_recs.append(r)

        return AgentOutput(
            agent_name = "RecommendationAgent",
            status     = "success",
            data = {
                "clinical_summary": summary,
                "summary_method":   method,
                "severity":         analysis_output.data.get("severity"),
                "ahi":              analysis_output.data.get("ahi"),
            },
            reasoning        = f"Summary generated using: {method}",
            recommendations  = unique_recs,
        )


# ── 5. ORCHESTRATOR AGENT ─────────────────────────────────────────────────────

class OrchestratorAgent:
    def __init__(self):
        self.data_agent            = DataAgent()
        self.analysis_agent        = AnalysisAgent()
        self.knowledge_agent       = KnowledgeRetrievalAgent()
        self.recommendation_agent  = RecommendationAgent()

    def run(self, raw_patient_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  Orchestrator: Processing {raw_patient_data.get('patient_id','?')}")
        print(f"{'='*60}")

        print("[1/4] DataAgent        -> Validating & enriching...")
        data_out = self.data_agent.process(raw_patient_data)
        print(f"      Status: {data_out.status}")

        print("[2/4] AnalysisAgent    -> Classifying severity...")
        analysis_out = self.analysis_agent.process(data_out)
        print(f"      Status: {analysis_out.status} | Severity: {analysis_out.data.get('severity','N/A')}")

        print("[3/4] KnowledgeAgent   -> Fetching guidelines...")
        knowledge_out = self.knowledge_agent.process(analysis_out)
        print(f"      Status: {knowledge_out.status}")

        print("[4/4] RecommendationAgent -> Synthesising report...")
        rec_out = self.recommendation_agent.process(data_out, analysis_out, knowledge_out)
        print(f"      Status: {rec_out.status}")

        report = {
            "patient_id":            raw_patient_data.get("patient_id"),
            "pipeline_status":       rec_out.status,
            "agents": {
                "DataAgent":               asdict(data_out),
                "AnalysisAgent":           asdict(analysis_out),
                "KnowledgeRetrievalAgent": asdict(knowledge_out),
                "RecommendationAgent":     asdict(rec_out),
            },
            "summary":               rec_out.data.get("clinical_summary", ""),
            "severity":              rec_out.data.get("severity"),
            "estimated_ahi":         rec_out.data.get("ahi"),
            "final_recommendations": rec_out.recommendations,
        }
        print(f"\n  Pipeline complete.")
        return report
