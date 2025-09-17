import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import time
import json

def run_domain_analysis(domain_name, vibration_data_domain, class_labels_domain, 
                       sampling_rate, window_size, decimation_factor, k_folds=5):
    """
    Run complete analysis for a single domain.
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYZING DOMAIN: {domain_name}")
    print(f"{'='*60}")
    
    # Create directory for this domain
    save_dir = f"../Models/domain_{domain_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Parameters
    input_shape = (int(2560/decimation_factor), 1)
    
    # Extract features for this domain
    print(f"Extracting features for {domain_name}...")
    X, Y = extract_features(vibration_data_domain, class_labels_domain)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Define k-Fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store metrics
    all_reports = []
    all_accuracies = []
    model_paths = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{k_folds}")
        
        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Build the model
        model = build_model(input_shape, num_classes=len(class_labels_domain))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        )
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(X_train, Y_train, epochs=100, batch_size=32, 
                          verbose=1, validation_data=(X_test, Y_test), 
                          callbacks=[early_stopping])
        
        # Save model
        model_path = os.path.join(save_dir, f"model_{domain_name}_fold_{fold+1}.h5")
        model.save(model_path)
        model_paths.append(model_path)
        print(f"Saved model: {model_path}")
        
        # Predict and evaluate
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(Y_test, axis=1)
        
        # Compute accuracy
        accuracy = np.mean(y_pred == y_true)
        all_accuracies.append(accuracy)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues",
                   xticklabels=class_labels_domain, yticklabels=class_labels_domain)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {domain_name} (Fold {fold+1})")
        
        cm_filename = os.path.join(save_dir, f"confusion_matrix_{domain_name}_fold_{fold+1}.png")
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store classification report with proper handling
        # Use target_names to ensure proper class names in report
        report = classification_report(y_true, y_pred, 
                                     target_names=class_labels_domain,
                                     output_dict=True, 
                                     zero_division=0)
        all_reports.append(report)
    
    # Calculate and save final results with proper error handling
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    
    # Create summary report with safe access to metrics
    summary_data = []
    metrics = ['precision', 'recall', 'f1-score']
    
    for label in class_labels_domain:  # Use actual class names, not indices
        row = []
        for metric in metrics:
            values = []
            for report in all_reports:
                # Access by actual class name
                if label in report and metric in report[label]:
                    values.append(report[label][metric])
                else:
                    values.append(0.0)  # Default value for missing metrics
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            row.append(f"{mean_val:.4f} ± {std_val:.4f}")
        
        # Add support with safe access
        support_vals = []
        for report in all_reports:
            if label in report and 'support' in report[label]:
                support_vals.append(report[label]['support'])
            else:
                support_vals.append(0)
        
        support_mean = int(np.mean(support_vals))
        row.append(support_mean)
        summary_data.append(row)
    
    # Create final summary DataFrame
    final_report_df = pd.DataFrame(summary_data, 
                                  columns=['Precision', 'Recall', 'F1-score', 'Support'], 
                                  index=class_labels_domain)
    
    # Save results
    final_report_csv = os.path.join(save_dir, f"final_report_{domain_name}.csv")
    final_report_df.to_csv(final_report_csv)
    
    # Save accuracy summary
    accuracy_summary = {
        'domain': domain_name,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_accuracies': all_accuracies,
        'model_paths': model_paths
    }
    
    accuracy_file = os.path.join(save_dir, f"accuracy_summary_{domain_name}.json")
    with open(accuracy_file, 'w') as f:
        json.dump(accuracy_summary, f, indent=2)
    
    print(f"\nDomain {domain_name} Results:")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Results saved to: {save_dir}")
    
    return accuracy_summary

def run_all_domains_analysis():
    """
    Run analysis for all three domains automatically.
    """
    # Domain configurations
    domains = [
        {
            'name': '0Nm',
            'data': vibration_data0,
            'labels': class_labels0
        },
        {
            'name': '2Nm', 
            'data': vibration_data1,
            'labels': class_labels1
        },
        {
            'name': '4Nm',
            'data': vibration_data2, 
            'labels': class_labels2
        }
    ]
    
    all_results = {}
    
    for domain in domains:
        try:
            results = run_domain_analysis(
                domain_name=domain['name'],
                vibration_data_domain=domain['data'],
                class_labels_domain=domain['labels'],
                sampling_rate=sampling_rate,
                window_size=window_size,
                decimation_factor=decimation_factor
            )
            all_results[domain['name']] = results
        except Exception as e:
            print(f"Error processing domain {domain['name']}: {str(e)}")
            continue
    
    # Create overall summary
    if all_results:
        create_overall_summary(all_results)
    
    return all_results

def create_overall_summary(all_results):
    """
    Create a summary comparing all domains.
    """
    summary_data = []
    for domain_name, results in all_results.items():
        summary_data.append({
            'Domain': domain_name,
            'Mean Accuracy': f"{results['mean_accuracy']:.4f}",
            'Std Accuracy': f"{results['std_accuracy']:.4f}",
            'Best Fold': f"{max(results['fold_accuracies']):.4f}",
            'Worst Fold': f"{min(results['fold_accuracies']):.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save overall summary
    overall_dir = "../Models/overall_summary"
    os.makedirs(overall_dir, exist_ok=True)
    
    summary_file = os.path.join(overall_dir, "domain_comparison_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    domains = list(all_results.keys())
    accuracies = [all_results[d]['mean_accuracy'] for d in domains]
    stds = [all_results[d]['std_accuracy'] for d in domains]
    
    plt.bar(domains, accuracies, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Mean Accuracy')
    plt.title('Domain Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    for i, (domain, acc) in enumerate(zip(domains, accuracies)):
        plt.text(i, acc + stds[i] + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(overall_dir, "domain_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nOverall summary saved to: {overall_dir}")
    print(f"Domain comparison summary:\n{summary_df}")


# Run the complete analysis
if __name__ == "__main__":
    all_results = run_all_domains_analysis()