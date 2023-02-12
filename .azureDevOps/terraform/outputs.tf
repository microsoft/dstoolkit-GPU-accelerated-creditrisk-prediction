output "id" {
  value = module.machine_learning_workspace.id
  sensitive = false
}

output "acr_name" {
  value = module.local_container_registry.name
  sensitive = false
}

output "acr_id" {
  value = module.local_container_registry.id
  sensitive = false
}

output "acr_login_server" {
  value = module.local_container_registry.login_server
  sensitive = false
}