# Credit Risk Scoring Resource Deployment

The following steps will deploy the resources required to run the credit risk scoring solution.  The resources will be deployed to an Azure subscription.  The following resources will be deployed:

* Azure Machine Learning Workspace
* Azure Machine Learning Compute Cluster
* Container Registry
* Azure Key Vault
* Azure Storage Account
* Azure Application Insights
* Resource Group

## Prerequisites

* Azure Subscription
* Azure DevOps Organization or Terraform locally installed
* Storage Account with a container named `tfstate` to store Terraform state
* Service Principal with Contributor access to the subscription (only required if using Azure DevOps)

A guide on how to create a Service Principal in DevOps can be found [here](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#sep-azure-rbac).

## Deploying Resources

Irrespective of whether you are using Azure DevOps or Terraform locally, you will need to make some changes to the configuration files. The following steps will guide you through the process.

In the directory `configuration` you need to make adjustments to the file `configuration-infra.variables.yml`.  The following variables need to be set:

* BACKEND_TERRAFROMSTATE_RG
* BACKEND_TERRAFROMSTATE_STORAGE_ACCOUNT
* BACKEND_TERRAFROMSTATE_CONTAINER_NAME
* TERRAFORMSTATE_FILE_NAME
* SERVICECONNECTION

In addition you can also change the BASENAME variable to a value of your choice.  This will be used as a prefix for all resources deployed. Furthermore, you can also change the location of the resources by changing the LOCATION variable. Lastly, you can also change the POSTFIX variable to a value of your choice.  This will be used as a suffix for all resources deployed (e.g. dev, test, prod).

After that you also need to make changes in the file `provider.tf`, which is located in the directory `.azureDevOps/terrafrom`.  In the backend section you need to change the following variables:

* resource_group_name = "BACKEND_TERRAFROMSTATE_RG"
* storage_account_name = "BACKEND_TERRAFROMSTATE_STORAGE_ACCOUNT"
* container_name = "BACKEND_TERRAFROMSTATE_CONTAINER_NAME"
* key = "TERRAFORMSTATE_FILE_NAME"

Once you have made the changes as instructed above, you can are ready to deploy the resources.

### Azure DevOps

If you are using Azure DevOps the resource deployment is as simple as running the pipeline which can be found in the directory `.azureDevOps/azure-pipelines/azure-pipeline.yml`.  The pipeline will deploy the resources to the subscription specified in the service connection.

### Terraform Locally

If you are using Terraform locally, you can run the following commands from the directory `.azureDevOps/terraform`:

```bash

terraform init

terraform plan

terraform apply

```

Both approaches will deploy the resources to the subscription specified in the service connection.

Next to the deployment of the resources the pipline will also create the docker environment in the Azure Machine Learning Workspace. If you are using Terraform locally, you can run the following commands from the directory `configuration/environment/docker`:

```bash

az login

az account set --subscription "SUBSCRIPTION_ID"

az ml environment create --file docker-context.yml

```

Note: In case you do not have the Azure Machine Learning CLI installed, you can install it as per the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli).
Further you can also create the enviroment manually in the Azure Machine Learning Studio. You can copy the contents of the docker file and paste it in the environment section of the Azure Machine Learning Studio.

