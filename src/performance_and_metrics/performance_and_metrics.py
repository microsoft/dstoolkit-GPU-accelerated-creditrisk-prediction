import os
import traceback

import matplotlib.pyplot as plt
import shap
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import plot_importance
import xgboost as xgb
import time
from utils import timeit
import numpy as np

class ClassificationReport:
    def __init__(self, run, nvidia_gpu_available, threshold) -> None:
        self.run = run
        self.nvidia_gpu_available = nvidia_gpu_available
        self.threshold = threshold

        self._asset_path = "/assets/"
        try:
            if not os.path.exists(self._asset_path):
                os.makedirs(self._asset_path)
        except Exception as e:
            print("Error in generating the folder to store assets.")
            traceback.print_exc()

    def log_ROC_image(self, actuals, probas):
        try:
            preds = (probas > self.threshold)
            fpr, tpr, _ = metrics.roc_curve(actuals, preds)
            roc_auc = metrics.auc(fpr, tpr)

            plt.rcParams["figure.figsize"] = [10,6]
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

            fig_path = os.path.join(self._asset_path, 'ROC.png')
            plt.savefig(fig_path, bbox_inches='tight')
            self.run.log_image(name='ROC', path=fig_path, plot=None, description='Receiver Operating Characteristics')
        except Exception as e:
            print("Error in logging ROC curve")
            traceback.print_exc()
    @timeit
    def generate_and_log_shap_plot(self, model, X):
        try:
            #if self.nvidia_gpu_available:
            #    model.set_param({"predictor": "gpu_predictor"})
            #else:
            #    model.set_param({"predictor": "cpu_predictor"})

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            print("Shape of SHAP values:", shap_values.shape)
            
            plt.figure()
            if self.nvidia_gpu_available: 
                shap_plot = shap.summary_plot(shap_values, X.to_pandas(),show=False)
            else: 
                shap_plot = shap.summary_plot(shap_values, X,show=False)
            
            shap_plot_path = os.path.join(self._asset_path, 'shap.png')
            plt.savefig(shap_plot_path, bbox_inches='tight')
            self.run.log_image(name='SHAP', path=shap_plot_path, plot=None, description='SHAP')
            return shap_values, explainer
        except Exception as e:
            print("Error in generating/logging SHAP plot/values.")
            traceback.print_exc()
            return (None, None)
    
    def log_shap_scatter_plot(self, shap_values, X):
        try:
            plt.figure()
            shap_scatter_plot = shap.dependence_plot(0, shap_values, X)
            #shap_scatter_plot = shap.plots.scatter(shap_values.iloc[:,0])            
            shap_scatter_plot_path = os.path.join(self._asset_path, 'shap_scatter.png')
            plt.savefig(shap_scatter_plot_path, bbox_inches='tight')
            self.run.log_image(name='SHAP_Scatter', path=shap_scatter_plot_path, plot=None, description='SHAP Scatter Plot')
        except Exception as e:
            print("Error in generating/logging SHAP scatter plot")
            traceback.print_exc()


    def log_importance_plot(self, model, feature_names):
        try:
            importance_fig_path = os.path.join(self._asset_path, 'importance.png')
            fig, ax = plt.subplots( nrows=1, ncols=1)
            x = plt.rcParams["figure.figsize"] = [10,6]
            ax = plot_importance(model, max_num_features=10)
            ax.figure.tight_layout()
            ax.figure.savefig(importance_fig_path)
            self.run.log_image(name='Importance Scores', path=importance_fig_path, plot=None, description='XGBoost Importance Scores')
        except Exception as e:
            print("Error in logging importance score plot")
            traceback.print_exc()



        
    def log_single_dependence_plots(self,shap_values ,feature,index,X): 

        try:
                start = time.time()
                plt.figure()
                shap.dependence_plot(feature, shap_values, X.to_pandas(),interaction_index = index)
                shap_plot_path =  os.path.join(self._asset_path, 'shap_dependence_feature_{}.png'.format(feature))
                plt.savefig(shap_plot_path, bbox_inches='tight')
                self.run.log_image(name='shap_dependence_{}_index_{}'.format(feature,index), path=shap_plot_path, plot=None, description='SHAP shap dependence plot for feature {} with interaction index {}'.format(feature, index))
                print(round(time.time()-start, 3),'secs time for SHAP dependence plot')   
        except Exception as e:
            print("Error in generating/logging SHAP dependence plot")

    

    def log_dependence_plots(self,shap_values ,number_important_feat,X): 

            try:
                # Compute absolute SHAP values
                abs_shap_values = np.abs(shap_values)

                # Compute feature importance scores by summing SHAP values over all samples
                feat_importance_scores = np.sum(abs_shap_values, axis=0)

                #Get the top "number_important_feat" features by importance
                top_feats = np.argsort(feat_importance_scores)[::-1][:number_important_feat]
                i= 0
                for feat in top_feats:
                    start = time.time()
                    plt.figure()
                    shap.dependence_plot(feat, shap_values, X.to_pandas())
                    shap_plot_path =  os.path.join(self._asset_path, 'shap_dependence_imp_feature_{}.png'.format(i))
                    plt.savefig(shap_plot_path, bbox_inches='tight')
                    self.run.log_image(name='shap_dependence_imp_feature_{}'.format(feat), path=shap_plot_path, plot=None, description='SHAP shap dependence plot for important feature number {}'.format(feat))
                    print(round(time.time()-start, 3),'secs time for SHAP dependence plot')   
                    i = i+1 
            except Exception as e:
                print("Error in generating/logging SHAP dependence plot")


    # Force plot for a given row in the dataframe
    def log_force_plot(self, model,row,X) : 
        try: 
            expl = shap.TreeExplainer(model)
            shap_values = expl.shap_values(X)
            start = time.time()
            plt.figure()
            shap.force_plot(expl.expected_value, shap_values[row,:], X.to_pandas().iloc[row,:],show=False,matplotlib=True)
            shap_plot_path = os.path.join(self._asset_path, 'shapforce_row_{}.png'.format(row))
            plt.savefig(shap_plot_path, bbox_inches='tight')
            self.run.log_image(name='SHAP Force row {}'.format(row), path=shap_plot_path, plot=None, description='SHAP force plot for observation {}'.format(row))
            print(round(time.time()-start, 3),'secs time for SHAP force plot') 
            
        except Exception as e:
            print("Error in generating/logging SHAP force plot")


    
    # Decision plots for first n observations

    def log_decision_plot(self, model,X,n):

        try :  
            expl = shap.TreeExplainer(model)
            shap_values = expl.shap_values(X)

            start = time.time()
            plt.figure()
            shap.initjs()
            shap.decision_plot(expl.expected_value, shap_values[0:n],feature_names=list(X.to_pandas().columns))
            shap_plot_path = os.path.join(self._asset_path, 'shapdecision.png')
            plt.savefig(shap_plot_path, bbox_inches='tight')
            self.run.log_image(name='SHAP decision', path=shap_plot_path, plot=None, description='SHAP decision plot')
            print(round(time.time()-start, 3),'secs time for SHAP decision plot')

        except Exception as e:
            print("Error in generating/logging SHAP decision plot")



    # waterfall plot for a given row in the dataframe 
    def log_waterfall_plot(self,model,shap_values,X,row): 
        try:
            start = time.time()
            plt.figure()
            shap_exp = shap.Explanation(values=shap_values, base_values=model.predict(xgb.DMatrix(X)), feature_names=X.columns)
            shap.waterfall_plot(shap_exp[row], max_display=10, show=False)
            shap_plot_path = os.path.join(self._asset_path, 'shap_waterfall_row_{}.png'.format(row))
            plt.savefig(shap_plot_path, bbox_inches='tight')
            self.run.log_image(name='SHAP_Waterfall {}'.format(row), path=shap_plot_path, plot=None, description='SHAP Waterfall Plot for observation {}'.format(row))
            print(round(time.time()-start, 3),'secs time for SHAP waterfall plot')   
        except Exception as e:
            print("Error in generating/logging SHAP waterfall plot")





    def _fetch_metrics(self, y_true, predictions):
        classification_report = classification_report(y_true, predictions)
        
        fpr, tpr, _ = metrics.roc_curve(y_true, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        return classification_report,roc_auc
    
    def get_metrics(self, y_true, probas):
        """
        """
        predictions = (probas > self.threshold)
        classification_report, roc_auc = self._fetch_metrics(y_true, predictions)
        return classification_report, roc_auc