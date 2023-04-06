module "machine_learning_workspace" {
  source = "./machine-learning-workspace/"

  basename = "${var.name}-${var.postfix}"
  rg_name = module.local_rg.name
  location = var.location

  storage_account_id      = module.local_storage_account.id
  key_vault_id            = module.local_key_vault.id
  application_insights_id = module.local_application_insights.id
  container_registry_id   = module.local_container_registry.id

  is_sec_module = false

  tags = {}
}

# Resource Group

module "local_rg" {
  source = "./resource-group/"

  basename = "${var.name}-${var.postfix}"
  location = var.location

  tags = local.tags
  
}

# Storage Account

module "local_storage_account" {
  source = "./storage-account/"

  basename = "${var.name}-${var.postfix}"
  rg_name = module.local_rg.name
  location = var.location

  hns_enabled = false
  firewall_default_action = "Allow"

  is_sec_module = false
}

# Key Vault

module "local_key_vault" {
  source = "./key-vault/"

  basename = "${var.name}-${var.postfix}-${random_string.postfix.result}"
  rg_name = module.local_rg.name
  location = var.location

  is_sec_module = false
}

# Application Insights

module "local_application_insights" {
  source = "./application-insights/"

  basename = "${var.name}-${var.postfix}"
  rg_name = module.local_rg.name
  location = var.location
}

# Azure Container Registry

module "local_container_registry" {
  source = "./container-registry/"

  basename = "${var.name}-${var.postfix}-${random_string.postfix.result}"
  rg_name = module.local_rg.name
  location = var.location

  is_sec_module = false
}

# Cluster

module "aml_cluster" {
  source = "./machine-learning-compute-cluster/"

  location = var.location
  machine_learning_workspace_id = module.machine_learning_workspace.id 
}