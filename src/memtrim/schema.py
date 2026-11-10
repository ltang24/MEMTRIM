from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidenceUnit:
    entity: str
    attribute: str
    value_type: str
    value: str
    span: str
    multi_valued: bool
    key_id: Optional[str] = None
    value_id: Optional[str] = None


@dataclass
class TaskSignature:
    operation: str
    target: str
    task_id: Optional[str] = None


@dataclass
class ParsedRecord:
    source_text: str
    evidence: List[EvidenceUnit] = field(default_factory=list)
    task: Optional[TaskSignature] = None
    support_mask: List[bool] = field(default_factory=list)
    outcome_text: Optional[str] = None
    outcome_evidence: Optional[EvidenceUnit] = None
    parse_ok: bool = True


@dataclass
class StoredInteraction:
    interaction_id: str
    text: str
    relation: str
    subject_id: str
    item_id: str
    is_primary: bool


@dataclass
class Item:
    item_id: str
    edit_id: str
    condition: str
    relation: str
    subject_id: str
    current_fact: str
    question: str
    new_target_forms: set
    memory_text: str
    unchanged_facts: List[str]
    store_id: str = ""
