# https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/machine_learning_compute_cluster

resource "azurerm_machine_learning_compute_cluster" "adl_mlw_compute_cluster" {
  name                          = var.basename
  location                      = var.location
  vm_priority                   = var.vm_priority
  vm_size                       = var.vm_size
  machine_learning_workspace_id = var.machine_learning_workspace_id
  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 1
    scale_down_nodes_after_idle_duration = "PT600S" # 30 seconds
  }
  identity {
    type = "SystemAssigned"
  }

}