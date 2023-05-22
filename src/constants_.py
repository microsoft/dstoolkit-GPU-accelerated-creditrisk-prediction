from collections import OrderedDict

DATA_DOWNLOAD_URL = "http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2016.tgz"

ACQUISITION_COLS = [
    "LoanID",
    "Channel",
    "SellerName",
    "OrInterestRate",
    "OrUnpaidPrinc",
    "OrLoanTerm",
    "OrDate",
    "FirstPayment",
    "OrLTV",
    "OrCLTV",
    "NumBorrow",
    "DTIRat",
    "CreditScore",
    "FTHomeBuyer",
    "LoanPurpose",
    "PropertyType",
    "NumUnits",
    "OccStatus",
    "PropertyState",
    "Zip",
    "MortInsPerc",
    "ProductType",
    "CoCreditScore",
    "Extra",
    "MortInsType",
    "RelMortInd",
]

dtypesDict = OrderedDict(
    [
        ("LoanID", "int64"),
        ("Channel", "category"),
        ("SellerName", "category"),
        ("OrInterestRate", "float64"),
        ("OrUnpaidPrinc", "int64"),
        ("OrLoanTerm", "int64"),
        ("OrDate", "str"),
        ("FirstPayment", "str"),
        ("OrLTV", "int64"),
        ("OrCLTV", "float64"),
        ("NumBorrow", "float64"),
        ("DTIRat", "float64"),
        ("CreditScore", "float64"),
        ("FTHomeBuyer", "category"),
        ("LoanPurpose", "category"),
        ("PropertyType", "category"),
        ("NumUnits", "int64"),
        ("OccStatus", "category"),
        ("PropertyState", "category"),
        ("Zip", "int64"),
        ("MortInsPerc", "float64"),
        ("ProductType", "category"),
        ("CoCreditScore", "float64"),
        ("Extra", "int64"),
        ("MortInsType", "category"),
        ("RelMortInd", "category"),
    ]
)
PERFORMANCE_COLS = [
    "LoanID",
    "MonthRep",
    "Servicer",
    "CurrInterestRate",
    "CAUPB",
    "LoanAge",
    "MonthsToMaturity",
    "AdMonthsToMaturity",
    "MaturityDate",
    "MSA",
    "CLDS",
    "ModFlag",
    "ZeroBalCode",
    "ZeroBalDate",
    "LastInstallDate",
    "ForeclosureDate",
    "DispositionDate",
    "ForeclosureCosts",
    "PPRC",
    "AssetRecCost",
    "MHRC",
    "ATFHP",
    "NetSaleProceeds",
    "CreditEnhProceeds",
    "RPMWP",
    "OFP",
    "NIBUPB",
    "PFUPB",
    "RMWPF",
    "FPWA",
    "ServicingIndicator",
]

DATASTORE_FOLDER_NAME = "credit_risk_data/"
BACKUP_DATA_FOLDER_NAME = "./data"


# dataset specific
TARGET_COL_NAME = "Default"

ID_COL_NAME = "LoanID"
