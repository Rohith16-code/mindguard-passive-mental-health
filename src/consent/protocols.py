"""IRB-compliant consent flow definitions."""
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator


class ConsentStatus(Enum):
    """Consent status enumeration."""
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"


class ProtocolVersion(BaseModel):
    """Protocol version metadata."""
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    effective_date: datetime
    expiry_date: Optional[datetime] = None
    irb_approval_number: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class ConsentSection(BaseModel):
    """A single section of the consent document."""
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    required: bool = True


class ConsentFlow(BaseModel):
    """Represents an IRB-compliant consent flow."""
    protocol: ProtocolVersion
    sections: List[ConsentSection]
    minimum_age: int = 18
    requires_electronic_signature: bool = True
    requires_review: bool = True
    data_use_terms: Dict[str, str] = Field(default_factory=dict)

    @validator("sections")
    def validate_sections(cls, v: List[ConsentSection]) -> List[ConsentSection]:
        if not v:
            raise ValueError("At least one consent section is required")
        required_titles = {"Introduction", "Risks and Benefits", "Privacy", "Voluntary Participation"}
        found_titles = {s.title for s in v}
        missing = required_titles - found_titles
        if missing:
            raise ValueError(f"Missing required sections: {missing}")
        return v

    def is_current(self, at_time: Optional[datetime] = None) -> bool:
        """Check if protocol is currently valid."""
        if at_time is None:
            at_time = datetime.now(timezone.utc)
        if self.protocol.expiry_date and at_time > self.protocol.expiry_date:
            return False
        return at_time >= self.protocol.effective_date

    def get_required_sections(self) -> List[ConsentSection]:
        """Return only required sections."""
        return [s for s in self.sections if s.required]


class ConsentRecord(BaseModel):
    """Record of a user's consent action."""
    user_id: str = Field(..., min_length=1)
    protocol_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    status: ConsentStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    signature_data: Optional[Dict[str, str]] = None
    review_status: Optional[str] = None
    review_timestamp: Optional[datetime] = None

    @validator("timestamp", pre=True, always=True)
    def set_timestamp(cls, v: Optional[datetime]) -> datetime:
        if v is None:
            return datetime.now(timezone.utc)
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @validator("review_timestamp", pre=True, always=True)
    def set_review_timestamp(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ConsentRequest(BaseModel):
    """Request to grant or update consent."""
    user_id: str = Field(..., min_length=1)
    protocol_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    sections_accepted: List[str] = Field(default_factory=list)
    signature_data: Optional[Dict[str, str]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ConsentResponse(BaseModel):
    """Response to a consent request."""
    success: bool
    consent_record: Optional[ConsentRecord] = None
    message: Optional[str] = None
    next_steps: Optional[List[str]] = None

    @validator("next_steps")
    def validate_next_steps(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        if len(v) == 0:
            return None
        return v


def validate_consent_request(
    request: ConsentRequest,
    protocol: ConsentFlow
) -> Tuple[bool, Optional[str]]:
    """Validate a consent request against the protocol."""
    if not protocol.is_current():
        return False, "Protocol is no longer active"
    if request.protocol_version != protocol.protocol.version:
        return False, f"Protocol version mismatch: expected {protocol.protocol.version}, got {request.protocol_version}"
    required_titles = {s.title for s in protocol.get_required_sections()}
    if not required_titles.issubset(set(request.sections_accepted)):
        missing = required_titles - set(request.sections_accepted)
        return False, f"Missing required section acceptances: {missing}"
    if protocol.requires_electronic_signature and not request.signature_data:
        return False, "Electronic signature required"
    return True, None


def create_consent_record(
    request: ConsentRequest,
    status: ConsentStatus,
    review_status: Optional[str] = None
) -> ConsentRecord:
    """Create a consent record from a request."""
    return ConsentRecord(
        user_id=request.user_id,
        protocol_version=request.protocol_version,
        status=status,
        ip_address=request.ip_address,
        user_agent=request.user_agent,
        signature_data=request.signature_data,
        review_status=review_status,
        review_timestamp=datetime.now(timezone.utc) if review_status else None
    )