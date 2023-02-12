locals {
  tags = {
    Project = "Credit-Risk-Scoring"
    Module = "azure-machine-learning-workspace"
    Toolkit = "Terraform"
    Environment = "${var.postfix}"
  }
}