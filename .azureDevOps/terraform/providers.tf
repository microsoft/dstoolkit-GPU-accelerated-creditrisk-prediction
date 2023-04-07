terraform {
  backend "azurerm" {
    resource_group_name  = "<insert-rg-name-here>"
    storage_account_name = "<insert-storage-account-name-here>"
    container_name       = "<insert-container-name-here>"
    key                  = "<insert-key-here>"
  }

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "= 3.42.0"
    }
  }
}
provider "azurerm" {
   features {}
}