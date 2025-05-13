import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from rtdl import FTTransformer

# Suppress warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

try:
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
except:
    st.write("Style file not found. Using default styling.")

# Title Section
st.title("💳 Credit Risk Prediction using PySpark, ML Models and FT-Transformer")
st.markdown(""" This project uses Machine Learning and Deep Learning (FT-Transformer) models to predict credit card approvals.
""")

# Sidebar for Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Project Overview", "Dataset", "Run Models", "Results"])

# Sidebar for Model Selection
model_option = st.sidebar.selectbox(
    'Select Model to Run',
    ('Traditional ML Models', 'FT-Transformer')
)

# Pre-load your fixed dataset
@st.cache_data
def load_data():
    data = pd.read_csv("AER_credit_card_data.csv")
    return data

df = load_data()

# Process data function for Traditional ML Models
def process_data(df):
    # Defining Columns
    categorical_cols = ['owner', 'selfemp']
    continuous_cols = ['reports', 'age', 'income', 'share', 'expenditure', 'dependents', 'months', 'majorcards', 'active']
    target_col = "card"
    
    # Handle Missing Values for Continuous Columns
    for col_name in continuous_cols:
        df[col_name].fillna(df[col_name].mean(), inplace=True)
    
    # Handle Missing Values for Categorical Columns
    for col_name in categorical_cols:
        df[col_name].fillna(df[col_name].mode()[0], inplace=True)
    
    # Converting Categorical columns to Numeric
    for col_name in categorical_cols:
        df[f"{col_name}_index"] = pd.Categorical(df[col_name]).codes
    
    feature_cols = [f"{c}_index" for c in categorical_cols] + continuous_cols
    X = df[feature_cols].values
    y = pd.Categorical(df[target_col]).codes
    
    return X, y, feature_cols

# Process data function for FT-Transformer
def process_data_for_transformer(df):
    categorical_cols = ['owner', 'selfemp']
    continuous_cols = ['reports', 'age', 'income', 'share', 'expenditure', 'dependents', 'months', 'majorcards', 'active']
    target_col = "card"
    
    for col in continuous_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode Categorical Columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Encode Target
    df[target_col] = LabelEncoder().fit_transform(df[target_col])

    X = df[categorical_cols + continuous_cols].values.astype(np.float32)
    y = df[target_col].values.astype(int)
    
    return X, y, categorical_cols + continuous_cols

# Train Traditional ML Models Function
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Base Models
    base_models = [
        ('CatBoost', CatBoostClassifier(verbose=0, random_state=42)),
        ('LightGBM', LGBMClassifier(random_state=42)),
        ('HistGradientBoost', HistGradientBoostingClassifier(random_state=42)),
        ('XGBoost', XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    
    # Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=base_models, 
        final_estimator=SklearnLogisticRegression(),
        cv=5
    )
    
    # Train Models
    trained_models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(base_models):
        status_text.text(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        progress_bar.progress((i + 1) / (len(base_models) + 1))
    
    status_text.text("Training Stacking Model...")
    stacking_model.fit(X_train, y_train)
    trained_models["Stacking Model"] = stacking_model
    progress_bar.progress(1.0)
    status_text.text("Training Complete!")
    
    return trained_models, X_train, X_test, y_train, y_test

# Train FT-Transformer Model
def train_transformer(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Add Gaussian Noise (Data Augmentation)
    X_train += torch.randn_like(X_train) * 0.01

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # Define FT-Transformer Model
    model = FTTransformer.make_baseline(
        n_num_features=X_train.shape[1],
        cat_cardinalities=None,
        n_blocks=8,
        d_token=128,
        attention_dropout=0.2,
        ffn_d_hidden=256,
        ffn_dropout=0.2,
        residual_dropout=0.2,
        d_out=2  # Binary classification
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=10, steps_per_epoch=len(train_loader))

    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    # For tracking metrics
    train_losses = []
    val_accuracies = []
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, None)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, None)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_accuracy = correct / total
        train_losses.append(total_loss/len(train_loader))
        val_accuracies.append(val_accuracy)
        
        # Update progress and status
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Plot training progress
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(train_losses)
        ax[0].set_title('Training Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        
        ax[1].plot(val_accuracies)
        ax[1].set_title('Validation Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        loss_chart.pyplot(fig)

    # Final Evaluation Metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    all_probs = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch, None)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    # Calculate Metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, all_probs),
        "Log Loss": log_loss(y_true, all_probs),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R-squared": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    }
    
    return model, metrics, (X_val, y_true, y_pred, all_probs), device

# Evaluate Models Function for Traditional ML
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(y_pred)
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Log Loss": log_loss(y_test, y_pred_proba) if hasattr(model, 'predict_proba') else "N/A",
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R-squared": r2_score(y_test, y_pred),
        "MAPE": np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
    }

# Sections
if section == "Project Overview":
    st.header("📚 Project Overview")
    st.write("""
    - **Dataset**: AER_credit_card_data.csv
    - **Goal**: Predict whether a credit card application will be approved or rejected.
    - **Models used**:
        - Traditional Machine Learning: Stacking, XGBoost, CatBoost, Random Forest, LightGBM, Histogram-based Gradient Boosting
        - Deep Learning: FT-Transformer
    """)
    st.image("/Users/adityadubey/Desktop/frontend/Image.jpg", width=800)

elif section == "Dataset":
    st.header("🗂️ Dataset Preview")
    st.write(df.head(20))
    st.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    st.subheader("Data Distribution")
    
    st.write("Numerical Feature Distributions")
    numerical_cols = ['reports', 'age', 'income', 'share', 'expenditure', 'dependents', 'months', 'majorcards', 'active']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f"Distribution of {col}")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("Target Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='card', data=df, ax=ax)
    ax.set_title("Distribution of Credit Card Approval")
    st.pyplot(fig)
    
    st.subheader("Feature Correlations")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Correlation Matrix")
    st.pyplot(fig)

elif section == "Run Models":
    st.header("🏃‍♂️ Running Models")
    
    if model_option == "Traditional ML Models":
        with st.spinner('Processing data and training models... This may take a few minutes...'):
            X, y, feature_cols = process_data(df)
            
            models, X_train, X_test, y_train, y_test = train_models(X, y)
            
            performance_results = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}
            
            st.session_state['performance_results'] = performance_results
            st.session_state['best_model'] = models["Stacking Model"]
            st.session_state['test_data'] = (X_test, y_test)
            st.session_state['models'] = models
            st.session_state['feature_cols'] = feature_cols
            
        st.success('Traditional ML Models Trained Successfully!')
        st.balloons()
        
        st.subheader("📈 Model Performance Metrics")
        
        metrics_df = pd.DataFrame({
            "Model": list(performance_results.keys()),
            "Accuracy": [round(metrics["Accuracy"] * 100, 2) for metrics in performance_results.values()],
            "F1-Score": [round(metrics["F1-Score"], 3) for metrics in performance_results.values()],
            "Precision": [round(metrics["Precision"], 3) for metrics in performance_results.values()],
            "Recall": [round(metrics["Recall"], 3) for metrics in performance_results.values()],
            "AUC": [round(metrics["AUC"], 3) for metrics in performance_results.values()]
        })
        
        st.table(metrics_df)
        
        # Highlight Best Performing Model
        best_model = metrics_df.iloc[metrics_df["Accuracy"].argmax()]["Model"]
        st.info(f"Best performing model: **{best_model}** with accuracy of {metrics_df.iloc[metrics_df['Accuracy'].argmax()]['Accuracy']}%")
        
        if best_model != "Stacking Model" and hasattr(models[best_model], 'feature_importances_'):
            st.subheader(f"Feature Importance for {best_model}")
            importances = models[best_model].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importances[indices][:10], y=[feature_cols[i] for i in indices][:10], ax=ax)
            ax.set_title(f"Top 10 Feature Importances - {best_model}")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
            
    elif model_option == "FT-Transformer":
        with st.spinner('Processing data and training FT-Transformer... This may take several minutes...'):
            try:
                X, y, feature_cols = process_data_for_transformer(df)
                
                model, metrics, eval_data, device = train_transformer(X, y)
                
                st.session_state['ft_metrics'] = metrics
                st.session_state['ft_eval_data'] = eval_data
                st.session_state['ft_model'] = model
                st.session_state['ft_device'] = device
                
                st.success('FT-Transformer Trained Successfully!')
                st.snow()
                
                st.subheader("📈 Model Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                with col3:
                    st.metric("AUC", f"{metrics['AUC']:.4f}")
                    st.metric("Log Loss", f"{metrics['Log Loss']:.4f}")
                
               
                st.subheader("Confusion Matrix")
                X_val, y_true, y_pred, _ = eval_data
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Rejected", "Approved"],
                            yticklabels=["Rejected", "Approved"], ax=ax)
                ax.set_title("Confusion Matrix for FT-Transformer")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error training FT-Transformer: {e}")
                st.warning("Please ensure you have the RTDL library installed: `pip install rtdl`")
                st.info("If you don't have the required libraries, you can try the Traditional ML Models instead.")

elif section == "Results":
    st.header("📊 Results and Visualizations")
    
    tab1, tab2 = st.tabs(["Traditional ML Models", "FT-Transformer"])
    
    with tab1:
        if 'performance_results' in st.session_state and 'best_model' in st.session_state and 'test_data' in st.session_state:
            performance_results = st.session_state['performance_results']
            best_model = st.session_state['best_model']
            X_test, y_test = st.session_state['test_data']
            
            st.subheader("Model Comparison (Accuracy)")
            
            accuracy_df = pd.DataFrame({
                "Model": list(performance_results.keys()),
                "Accuracy": [metrics["Accuracy"] * 100 for metrics in performance_results.values()]
            }).set_index("Model")
            
            st.bar_chart(accuracy_df)
            
            metrics_to_compare = ["Precision", "Recall", "F1-Score", "AUC"]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics_to_compare):
                metric_df = pd.DataFrame({
                    "Model": list(performance_results.keys()),
                    metric: [metrics[metric] for metrics in performance_results.values()]
                })
                
                sns.barplot(x="Model", y=metric, data=metric_df, ax=axes[i])
                axes[i].set_title(f"Model Comparison - {metric}")
                axes[i].tick_params(axis='x', rotation=45)
                
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Confusion Matrix (CatBoost)")
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Rejected", "Approved"],
                        yticklabels=["Rejected", "Approved"], ax=ax)
            ax.set_title("Confusion Matrix for CatBoost")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
            
            st.subheader("ROC Curves")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for name, model in [("Stacking Model", best_model)] + [(name, st.session_state['models'][name]) for name in ["CatBoost", "XGBoost", "Random Forest"] if 'models' in st.session_state]:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            st.pyplot(fig)
            
        else:
            st.warning("Please run Traditional ML models first to see results.")
            st.info("Go to the 'Run Models' section and select 'Traditional ML Models'.")
    
    with tab2:
        if 'ft_metrics' in st.session_state and 'ft_eval_data' in st.session_state:
            metrics = st.session_state['ft_metrics']
            X_val, y_true, y_pred, y_proba = st.session_state['ft_eval_data']
            
            st.subheader("FT-Transformer Performance")
            
            metrics_df = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Value": [f"{value:.4f}" if isinstance(value, (int, float)) else value for value in metrics.values()]
            })
            
            st.table(metrics_df)
            
            st.subheader("ROC Curve for FT-Transformer")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'FT-Transformer (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve - FT-Transformer')
            ax.legend(loc="lower right")
            st.pyplot(fig)
            
        else:
            st.warning("Please run FT-Transformer first to see results.")
            st.info("Go to the 'Run Models' section and select 'FT-Transformer'.")
            