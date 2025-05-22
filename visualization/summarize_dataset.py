import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# ===== Configuration =====
final_data_folder = "../data/final"
clinical_data_path = os.path.join(final_data_folder, "clinical.xlsx")
output_directory = "summary_output"
os.makedirs(output_directory, exist_ok=True)

# ===== Style Settings =====
plt.style.use('seaborn-whitegrid')
sns.set_palette("colorblind")  # Better for color vision deficiencies
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlepad'] = 15

# Simple, professional color palette
plot_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"]

# ===== Helper Functions =====
def convert_to_numeric(column):
    """Safely convert column to numeric, handling errors"""
    return pd.to_numeric(df[column].astype(str), errors="coerce")

def save_plot(figure, filename, dpi=300):
    """Save figure with consistent settings"""
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_directory, filename), 
        dpi=dpi, 
        bbox_inches='tight',
        transparent=False
    )
    plt.close()

def format_excel_sheet(writer, sheet_name, dataframe):
    """Apply professional formatting to Excel sheet with NaN handling"""
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # Header formatting
    header_style = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#4F81BD',
        'font_color': 'white',
        'border': 1,
        'font_name': 'Calibri'
    })
    
    # Data formatting (with NaN handling)
    data_style = workbook.add_format({
        'font_name': 'Calibri',
        'border': 1,
        'valign': 'top'
    })
    
    # NaN formatting
    nan_style = workbook.add_format({
        'font_name': 'Calibri',
        'border': 1,
        'valign': 'top',
        'font_color': '#999999'
    })
    
    # Write header
    for col_num, column_name in enumerate(dataframe.columns):
        worksheet.write(0, col_num, column_name, header_style)
    
    # Write data with NaN handling
    for row_num in range(len(dataframe)):
        for col_num, value in enumerate(dataframe.iloc[row_num]):
            if pd.isna(value) or value in [np.inf, -np.inf]:
                worksheet.write(row_num + 1, col_num, "N/A", nan_style)
            else:
                worksheet.write(row_num + 1, col_num, value, data_style)
    
    # Set column widths
    for idx, column in enumerate(dataframe.columns):
        max_len = max((
            dataframe[column].astype(str).map(len).max(),
            len(str(column))
        )) + 2
        worksheet.set_column(idx, idx, min(max_len, 35))
    
    # Add filters and freeze panes
    worksheet.autofilter(0, 0, 0, len(dataframe.columns) - 1)
    worksheet.freeze_panes(1, 0)

# ===== Data Loading =====
print("Loading and preparing data...")
df = pd.read_excel(clinical_data_path)
df["Inclusion"] = df["Inclusion"].astype(int)
df = df[df["Inclusion"] == 1]

# Filter to only patients with existing folders
available_patients = set(os.listdir(final_data_folder))
df = df[df["Study number"].astype(str).isin(available_patients)]

# Clean numeric columns
numeric_columns = [
    "Age at time of inclusion",
    "Size max entrance (cm)",
    "Time to progression in months",
    "Time to treatment in months",
    "Total follow-up duration in months"
]

for col in numeric_columns:
    df[col] = convert_to_numeric(col)

# Replace infinities with NaN and then fill NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ===== Analysis Functions =====
def create_summary_tables():
    """Generate well-organized summary tables by country"""
    summary_results = {}
    grouped_data = df.groupby("Country")
    
    # Categorical variables analysis
    categorical_vars = [
        "Gender", "Anatomical site", "Beta catenin", "PD (yes, no)",
        "Treatment during FU", "Type of treatment", 
        "Status at last follow up", 
        "Pregnancy during active surveillance (yes/no)"
    ]
    
    for var in categorical_vars:
        if var in df.columns:
            result_table = (grouped_data[var].value_counts(normalize=True)
                          .mul(100).round(1)
                          .unstack()
                          .fillna(0)
                          .replace([np.inf, -np.inf], np.nan)
                          .astype(str) + '%').T
            summary_results[f"CAT_{var}"] = result_table
    
    # Numerical variables analysis
    numerical_vars = [
        "Age at time of inclusion",
        "Size max entrance (cm)",
        "Time to progression in months",
        "Time to treatment in months",
        "Total follow-up duration in months"
    ]
    
    for var in numerical_vars:
        if var in df.columns:
            stats = grouped_data[var].agg(["count", "mean", "std", "median", "min", "max"])
            stats = stats.replace([np.inf, -np.inf], np.nan)
            stats["mean"] = stats["mean"].round(2)
            stats["std"] = stats["std"].round(2)
            summary_results[f"NUM_{var}"] = stats
    
    return summary_results

# Generate summary tables
summary_tables = create_summary_tables()

# Save to Excel with improved organization and NaN handling
print("Creating Excel report...")
with pd.ExcelWriter(
    os.path.join(output_directory, "clinical_summary.xlsx"), 
    engine='xlsxwriter',
    engine_kwargs={'options': {'nan_inf_to_errors': True}}
) as excel_writer:
    
    # Group related sheets together
    for sheet_type in ["CAT", "NUM"]:
        for sheet_name, table in summary_tables.items():
            if sheet_name.startswith(sheet_type):
                clean_name = sheet_name.replace(sheet_type + "_", "")[:25]
                table.to_excel(excel_writer, sheet_name=f"{sheet_type}_{clean_name}")
                format_excel_sheet(excel_writer, f"{sheet_type}_{clean_name}", table)
    
    # Add overview sheet
    overview_stats = pd.DataFrame({
        "Metric": [
            "Total Patients",
            "Countries Represented",
            "Average Age (years)",
            "Progression Disease Rate",
            "Average Follow-up (months)"
        ],
        "Value": [
            df["Study number"].nunique(),
            df["Country"].nunique(),
            df["Age at time of inclusion"].mean().round(1),
            f"{df['PD (yes, no)'].value_counts(normalize=True).get('yes', 0)*100:.1f}%",
            f"{df['Total follow-up duration in months'].mean():.1f}"
        ]
    })
    
    overview_stats.to_excel(excel_writer, sheet_name="Overview", index=False)
    format_excel_sheet(excel_writer, "Overview", overview_stats)

print("Summary tables saved to clinical_summary.xlsx")

# ===== Visualization Functions =====
def plot_disease_progression():
    """Create clean progression disease status plot"""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=df, 
        x="Country", 
        hue="PD (yes, no)", 
        palette=plot_colors,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.title("Disease Progression Status by Country", pad=15)
    plt.xlabel("Country")
    plt.ylabel("Patient Count")
    plt.legend(title="PD Status", frameon=True)
    
    # Add clean value labels
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{int(height)}',
                ha='center', 
                va='bottom',
                fontsize=9
            )
    
    save_plot(plt.gcf(), "disease_progression.png")

def plot_time_metrics():
    """Visualize time-based metrics with clean presentation"""
    time_vars = [
        "Time to progression in months",
        "Time to treatment in months",
        "Total follow-up duration in months"
    ]
    
    for time_var in time_vars:
        if time_var in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Boxplot with enhanced aesthetics
            box = sns.boxplot(
                data=df,
                x="Country",
                y=time_var,
                palette=plot_colors,
                width=0.6,
                showfliers=False,
                linewidth=1
            )
            
            # Add stripplot for individual points
            sns.stripplot(
                data=df,
                x="Country",
                y=time_var,
                color=".3",
                size=4,
                alpha=0.5,
                jitter=0.2
            )
            
            # Formatting
            clean_title = time_var.replace(" in months", "").title()
            plt.title(f"{clean_title} by Country", pad=15)
            plt.xlabel("Country")
            plt.ylabel("Months")
            
            # Rotate labels if needed
            if len(df["Country"].unique()) > 4:
                plt.xticks(rotation=30, ha='right')
            
            save_plot(plt.gcf(), f"{time_var.lower().replace(' ', '_')}.png")

def plot_patient_status():
    """Visualize patient status at last follow-up"""
    if "Status at last follow up" in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Order by frequency for better readability
        status_order = df["Status at last follow up"].value_counts().index
        
        ax = sns.countplot(
            data=df,
            x="Status at last follow up",
            hue="Country",
            order=status_order,
            palette=plot_colors,
            edgecolor='w',
            linewidth=0.5
        )
        
        plt.title("Patient Status at Last Follow-Up", pad=15)
        plt.xlabel("Status")
        plt.ylabel("Count")
        plt.legend(title="Country", frameon=True)
        plt.xticks(rotation=45, ha='right')
        
        save_plot(plt.gcf(), "patient_status.png")

def plot_imaging_sequences():
    """Visualize available imaging sequences by country"""
    sequence_counts = defaultdict(lambda: defaultdict(int))

    for _, patient in df.iterrows():
        patient_id = str(patient["Study number"])
        country = patient["Country"]
        patient_folder = os.path.join(final_data_folder, patient_id)

        if os.path.exists(patient_folder):
            try:
                for file in os.listdir(patient_folder):
                    if file.endswith(".nii.gz") and "-mask" not in file:
                        sequence = file.replace(".nii.gz", "")
                        sequence_counts[country][sequence] += 1
            except (PermissionError, FileNotFoundError) as e:
                print(f"Could not process {patient_folder}: {e}")
                continue

    if not sequence_counts:
        print("No imaging sequence data found")
        return

    # Prepare data for visualization
    seq_df = pd.DataFrame(sequence_counts).fillna(0).astype(int).T
    seq_df = seq_df[sorted(seq_df.columns)]  # Sort columns alphabetically
    
    # Calculate percentages
    total_counts = seq_df.sum(axis=1)
    percentage_df = seq_df.div(total_counts, axis=0) * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot counts
    seq_df.plot(kind='bar', stacked=False, ax=ax1, color=plot_colors)
    ax1.set_title("Imaging Sequence Counts by Country", pad=15)
    ax1.set_ylabel("Count")
    ax1.legend(title="Sequence", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Plot percentages
    percentage_df.plot(kind='bar', stacked=False, ax=ax2, color=plot_colors)
    ax2.set_title("Imaging Sequence Distribution (%) by Country", pad=15)
    ax2.set_ylabel("Percentage")
    ax2.legend().set_visible(False)
    
    plt.tight_layout()
    save_plot(fig, "imaging_sequences.png")

# Generate all visualizations
print("Creating visualizations...")
plot_disease_progression()
plot_time_metrics()
plot_patient_status()
plot_imaging_sequences()

print(f"Analysis complete. Results saved to: {output_directory}")