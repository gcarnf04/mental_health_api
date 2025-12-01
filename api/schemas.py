from pydantic import BaseModel
from typing import Dict, Optional

class DiagnosisRequest(BaseModel):
    patient_text: str = """
    Diagnostic Formulation and Clinical Impressions
    Presenting Problem and History:
    The patient presents with a chief complaint of persistent, debilitating low mood, anhedonia, and a significant decrease in overall functioning spanning at least the past six months, which has recently intensified. The patient reports a profound sense of hopelessness, pervasive fatigue (anergy), and difficulty initiating or completing even routine tasks. They describe their emotional state as "heavy" and "numb," distinct from temporary sadness. Historically, the patient reports a prior episode of similar, though less severe, symptoms approximately three years ago, which was untreated and resolved spontaneously after about four months, making this current presentation a recurrence. The current episode was not precipitated by a single, identifiable major stressor but rather by a gradual erosion of coping mechanisms following sustained professional and personal pressures.
    Symptom Checklist (DSM-5 Criteria A):
    Based on clinical interview and standardized screening tools (e.g., PHQ-9, Hamilton Depression Rating Scale), the patient endorses five (or more) of the following symptoms presenting nearly every day for most of the day, during the same 2-week period:
    Depressed mood most of the day, nearly every day, as indicated by subjective report (e.g., feeling sad, empty, or hopeless).
    Markedly diminished interest or pleasure in all, or almost all, activities most of the day, nearly every day (anhedonia). The patient has stopped engaging in previously enjoyed hobbies and reports food tastes "flat."
    Significant weight loss when not dieting, or weight gain, or decrease or increase in appetite nearly every day. The patient reports a 7-pound unintentional weight loss over the past 3 months due to a dramatically reduced appetite.
    Insomnia or hypersomnia nearly every day. The patient reports initial and middle insomnia, waking up frequently and having difficulty falling back asleep, resulting in unrefreshing sleep.
    Psychomotor agitation or retardation nearly every day (observable by others). Psychomotor retardation is noted during the session, characterized by slowed speech, decreased body movements, and long latencies before responding.
    Fatigue or loss of energy nearly every day (anergy), impacting work and social participation.
    Feelings of worthlessness or excessive or inappropriate guilt nearly every day. The patient expresses intense self-blame for their current state and perceived professional failures.
    Diminished ability to think or concentrate, or indecisiveness, nearly every day. The patient reports difficulties with memory retention and completing complex cognitive tasks.
    Recurrent thoughts of death (not just fear of dying), recurrent suicidal ideation without a specific plan, or a suicide attempt or a specific plan for committing suicide. The patient endorses passive suicidal ideation, specifically thoughts that they "would be better off not waking up," but denies an active plan or intent at this time.
    Differential Diagnosis and Exclusions (DSM-5 Criteria B, C, D, E):
    Criterion B (Distress/Impairment): The symptoms cause clinically significant distress and impairment in social, occupational, and other important areas of functioning (e.g., patient is on leave from work, avoiding all social contact).
    Criterion C (Substance/Medical): There is no evidence from the patient's history or collateral reports that the symptoms are attributable to the physiological effects of a substance (e.g., drug of abuse, medication) or another medical condition. Relevant medical workup (thyroid panel, complete blood count) is recommended to rule out physiological causes, but depression is the likely primary diagnosis.
    Criterion D (Manic/Hypomanic Episodes): There has never been a manic or hypomanic episode, ruling out Bipolar I or Bipolar II Disorder.
    Criterion E (Psychotic/Schizophrenia): The symptoms are not better explained by Schizoaffective Disorder, Schizophrenia, or other Psychotic Disorders. The patient denies hallucinations or delusions.
    Final Diagnosis (DSM-5)
    F33.2 Major Depressive Disorder, Recurrent Episode, Severe (Without Psychotic Features)
    Specifiers:
    Severity: Severe. This specifier is applied due to the number of symptoms endorsed (significantly greater than the required five) and the intensity of the symptoms (e.g., profound anhedonia, significant impairment in functioning, passive suicidal ideation), which render the patient barely able to function.
    In Partial/Full Remission: Not applicable. Currently in a full episode.
    With Anxious Distress: Mild. The patient reports some feelings of tension and restlessness, but this is not the predominant feature.
    With Melancholic Features: Strongly suggested. Features include profound anhedonia, lack of reactivity to usually pleasurable stimuli, early morning awakening, psychomotor retardation, excessive guilt, and anorexia/weight loss.
"""
    auto_classify: Optional[bool] = True
    pathology: Optional[str] = None

class ClassificationResult(BaseModel):
    pathology: str
    confidence: float
    all_probabilities: Dict[str, float]

class Metadata(BaseModel):
    original_text_length: int
    summary_length: int
    recommendation_length: int

class DiagnosisResponse(BaseModel):
    classification: ClassificationResult
    summary: str
    recommendation: str
    metadata: Metadata