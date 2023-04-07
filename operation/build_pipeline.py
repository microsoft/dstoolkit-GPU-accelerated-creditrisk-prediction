import os
import argparse
from azureml.data import OutputFileDatasetConfig
from azureml.core import Datastore, Dataset, Environment, Workspace
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
# msi authentication
from azureml.core.authentication import MsiAuthentication

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", type=str, dest="subscription_id", help="subscription id")
    parser.add_argument("--resource_group", type=str, dest="resource_group", help="resource group")
    parser.add_argument("--workspace_name", type=str, dest="workspace_name", help="workspace name", default="mlw-creditrisk-dev")
    parser.add_argument("--use_gpu", type=bool, dest="use_gpu", help="use gpu", default=True)
    parser.add_argument("--url", type=str, dest="url", help="data url", default="http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2016.tgz")
    parser.add_argument("--data_dir", type=str, dest="data_dir", help="data directory", default="data")
    parser.add_argument("--years", type=str, dest="years", help="years", default="2007,2008,2009")
    args = parser.parse_args()
    return args


def main():
    args = args_parser()

    # get the workspace 
    #msi_auth = MsiAuthentication()
    ws = Workspace.get(
        name=args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group)
       # auth=msi_auth)
    # get cluster
    if args.use_gpu:
        cluster_name =  'gpu-cluster'
    else:
        cluster_name = 'cpu-cluster'

    if cluster_name in ws.compute_targets:
        gpu_cluster = ws.compute_targets[cluster_name]
        if gpu_cluster and type(gpu_cluster) is AmlCompute:
            print('Found compute target. Will use {0} '.format(cluster_name))
    else:
        print('creating new cluster')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = 'Standard_NC12s_v3',  #'Standard_M128s'
                                                                    max_nodes = 1,
                                                                    idle_seconds_before_scaledown = 600,
                                                                    vm_priority = "lowpriority")
        
        gpu_cluster = ComputeTarget.create(ws, cluster_name, provisioning_config)
        gpu_cluster.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)


    env = Environment.get(workspace=ws, name="credit-risk")

    env.docker.enabled = True
    env.docker.base_image = None
    env.python.user_managed_dependencies = True
    env.environment_variables = None
    env.python.interpreter_path = "/opt/conda/envs/rapids/bin/python"

    run_config = RunConfiguration()
    run_config.environment = env
    # Create the pipeline steps

    # Step 1: Download the data

    data_download_step = PythonScriptStep(
        name="Download Data",
        source_directory="./src",
        script_name="data_download.py",
        compute_target=gpu_cluster,
        arguments=["--data-url", args.url, "--data_dir", args.data_dir],
        runconfig=run_config,
        allow_reuse=True
    )

    # Step 2: Run main (TODO: Here we could split the main in multiple steps)

    main_step = PythonScriptStep(
        name="Run Main",
        source_directory="./src",
        script_name="crs_main.py",
        compute_target=gpu_cluster,
        arguments=["--years", args.years],
        runconfig=run_config,
        
        allow_reuse=False
    )

    # Run the pipeline

    from azureml.pipeline.core import Pipeline, StepSequence
    steps = StepSequence(steps=[data_download_step, main_step])
    pipeline = Pipeline(workspace=ws, steps=steps)
    pipeline_run = pipeline.submit("credit-default-risk-sample")

    pipeline_run.wait_for_completion()

if __name__ == "__main__":
    main()





    
    


    





