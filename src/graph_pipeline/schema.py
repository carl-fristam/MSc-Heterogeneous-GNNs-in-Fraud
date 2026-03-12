from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetSchema:

    """
    Maps logical roles to column names in a transaction dataset.

    Required fields (must exist in every dataset):
        sender_id       — column identifying the sending account
        receiver_id     — column identifying the receiving account
        amount          — transaction amount
        timestamp       — date or datetime column
        label           — fraud/laundering label (can be None if labels come from a separate file)

    Optional fields (may be None if the dataset doesn't have them):
        timestamp_time  — separate time column (SAML-D has Date + Time separately)
        payment_currency, received_currency, sender_location, receiver_location, payment_type
        transaction_id, channel, device_id, customer_id, payment_method, destination_country
    """

    # ---- Required ----
    sender_id: str = "Sender_account"
    receiver_id: str = "Receiver_account"
    amount: str = "Amount"
    timestamp: str = "Date"
    label: Optional[str] = "Is_laundering"

    # ---- Time (SAML-D has date and time in separate columns) ----
    timestamp_time: Optional[str] = "Time"

    # ---- Categoricals ----
    payment_currency: Optional[str] = "Payment_currency"
    received_currency: Optional[str] = "Received_currency"
    sender_location: Optional[str] = "Sender_bank_location"
    receiver_location: Optional[str] = "Receiver_bank_location"
    payment_type: Optional[str] = "Payment_type"

    # ---- Optional (bank data has these, SAML-D doesn't) ----
    transaction_id: Optional[str] = None
    channel: Optional[str] = None
    device_id: Optional[str] = None
    customer_id: Optional[str] = None
    payment_method: Optional[str] = None
    destination_country: Optional[str] = None

SAML_D_SCHEMA = DatasetSchema()   # defaults match SAML-D column names

BANK_RETAIL_SCHEMA = DatasetSchema(
    sender_id="ACCOUNTID",
    receiver_id="COUNTERPARTYID",
    amount="BASEVALUE",
    timestamp="EVENTTIME",
    label=None,                    # labels come from a separate fraud_data_ret table
    timestamp_time=None,           # EVENTTIME is a full datetime
    payment_currency="CURRENCY",
    received_currency="BASECURRENCY",
    sender_location="ACCAGENTCOUNTRY",
    receiver_location="DESTINATIONCOUNTRY",
    payment_type="PAYMENTMETHOD",
    transaction_id="TRANSACTIONID",
    channel="CHANNEL",
    device_id="DEVICEID",
    customer_id="CUSTOMERID",
    payment_method="PAYMENTSUBMETHOD",
    destination_country="DESTINATIONCOUNTRY",
)

def get_schema(dataset_name: str) -> DatasetSchema:
    schemas = {
        "saml-d": SAML_D_SCHEMA,
        "bank-retail": BANK_RETAIL_SCHEMA,
    }
    if dataset_name not in schemas:
        raise ValueError(f"Unknown dataset name '{dataset_name}'. Valid options are: {list(schemas.keys())}")
    return schemas[dataset_name]