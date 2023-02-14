terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-backend"
    storage_account_name = "terraformbackendfr"
    container_name       = "terraform-dev"
    key                  = "tf/terraform.tfstate"
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