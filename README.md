# THE EMPATHY GAP

**An Experiment on How Perceptions Fuel Public Service Corruption**

A comprehensive statistical analysis pipeline for examining empathy gaps in academic dishonesty contexts, investigating how peer framing and authority framing influence unethical behavior and corruption.

## Overview

This analysis examines:
- **Copy likelihood** and **share likelihood** of academic solutions
- **Cheating behavior** (coin flip honesty task)
- **Expedite offers** and **expected favors** (in INR)
- **Language markers** (empathetic vs. impatient communication)
- **Interaction effects** between peer framing and TA (teaching assistant) framing

## Requirements

```bash
pip install numpy pandas matplotlib statsmodels scipy
```

**Python Version:** 3.8+

## Input Data

The script expects a CSV file named `empathygap-dataset.csv` with the following columns:

- `PEER FRAME` - Experimental peer framing condition (categorical)
- `TA FRAME` - Experimental TA framing condition (categorical)
- `HONOR CODE` - Honor code condition (categorical)
- `PROGRAM` - Academic program (categorical)
- `STRESS (0-100)` - Self-reported stress level
- `COPY LIKELIHOOD (0-100)` - Likelihood of copying solutions
- `SHARE LIKELIHOOD (0-100)` - Likelihood of sharing solutions
- `EXPEDITE OFFER (INR)` - Monetary offer to expedite help
- `EXPECTED FAVOR (INR)` - Expected monetary favor
- `COIN REPORT (HONESTY TASK)` - Reported coin flip result (Heads/Tails)
- `PEER HELPFULNESS (1-7)` - Perception of peer helpfulness
- `GRADER STRICTNESS (1-7)` - Perception of grader strictness
- `DECISION TIME (SECONDS)` - Time taken for decision
- `DM TO NIKHIL` - Direct message text for language analysis

## Usage

```bash
python empathy-gap.py
```

The script automatically:
1. Loads the dataset
2. Engineers features
3. Generates descriptive statistics
4. Creates visualizations
5. Runs statistical models
6. Exports results

## Output Structure

All outputs are saved to the `outputs/` directory:

### Tables (CSV + LaTeX)
- `descriptives_overall.csv` - Overall descriptive statistics
- `means_peer_ta.csv` - Means by PEER FRAME × TA FRAME
- `freq_*.csv` - Frequency tables for categorical variables
- `anova_*.csv` - ANOVA tables for OLS models
- `correlation_matrix.csv` - Correlation matrix
- `heads_rate_peer_ta.csv` - Cheating rates by condition
- `logit_cheat_params.csv` - Logistic regression coefficients
- `language_means_peer.csv` - Language marker means

### Visualizations
- `pie_*.png` - Distribution pie charts
- `hist-*.png` - Histograms for continuous variables
- `box-*.png` - Boxplots by experimental conditions
- `heatmap-*.png` - Annotated heatmaps for mean grids
- `bar_emp_mark_peer.png` - Empathy markers by peer frame
- `bar_imp_mark_peer.png` - Impatience markers by peer frame
- `correlation_heatmap.png` - Correlation heatmap

## Analysis Pipeline

### 1. Feature Engineering
- Converts INR categories to numeric amounts
- Creates binary cheating indicator (HEADS)
- Generates short aliases for continuous variables

### 2. Language Analysis
Extracts linguistic markers:
- **Empathy markers**: "sorry", "thanks", "appreciate", "understand", "we", "let's", "no worries"
- **Impatience markers**: "asap", "urgent", "right now", "immediately", "running out of time", "!!"

### 3. Descriptive Statistics
- Overall descriptive stats for all continuous variables
- Means by PEER FRAME × TA FRAME combinations
- Frequency distributions for categorical variables

### 4. Visualizations
- Histograms for continuous variables
- Boxplots by experimental conditions
- Annotated heatmaps showing mean patterns
- Bar charts for language markers

### 5. Statistical Models

#### OLS Regression Models
- **COPY**: Copy likelihood ~ PEER FRAME × TA FRAME + STRESS + HONOR CODE
- **SHARE**: Share likelihood ~ PEER FRAME × TA FRAME + STRESS + HONOR CODE
- **EXPEDITE_AMT**: Expedite amount ~ PEER FRAME × TA FRAME + STRESS + HONOR CODE

#### Logistic Regression
- **HEADS** (cheating): Logit model for coin flip dishonesty

#### Language Mechanism Models
- **EMP_MARK**: Empathy markers ~ PEER FRAME × TA FRAME
- **IMP_MARK**: Impatience markers ~ PEER FRAME × TA FRAME

### 6. Cheating Analysis
- Overall cheating rate (proportion reporting "Heads")
- Binomial test against expected 50% rate
- Cheating rates by experimental conditions

### 7. Correlation Analysis
- Correlation matrix for all key variables
- Visual correlation heatmap

## Key Features

- **Color-coded terminal output** for easy reading
- **Automatic directory creation** for outputs
- **Dual export format** (CSV + LaTeX) for academic writing
- **Comprehensive visualizations** for presentations
- **Interaction effects** testing between experimental conditions
- **Linguistic analysis** for mechanism exploration

## Statistical Methods

- **ANOVA**: Type II sum of squares for interaction effects
- **OLS Regression**: Linear models with categorical × categorical interactions
- **Logistic Regression**: Binary outcome modeling for cheating
- **Binomial Test**: Testing deviation from expected coin flip probability
- **Pearson Correlation**: Examining relationships between continuous variables

## Interpretation Notes

- **HEADS = 1**: Participant reported "Heads" (possible dishonesty indicator)
- **Higher COPY/SHARE**: Greater likelihood of academic dishonesty
- **EMP_MARK**: Count of empathetic language markers
- **IMP_MARK**: Count of impatient/urgent language markers
- **EXPEDITE_AMT**: Amount offered to expedite help (0-2000 INR)

## Customization

To modify the analysis:
- **Input file**: Change `inpt` variable (line 34)
- **Output directory**: Change `outs` variable (line 35)
- **Plot style**: Modify `plt.rcParams` settings (lines 37-38)
- **Language markers**: Edit `emwr` and `imwr` lists (lines 87-88)
- **Model formulas**: Adjust formula strings (lines 267-283)
