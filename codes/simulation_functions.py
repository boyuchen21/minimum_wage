import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from scipy.stats import gaussian_kde

# Function for data generating process
def generate_log_wages(n, μ=3, σ=1):
    wages = np.random.lognormal(mean=μ, sigma=σ, size=n)
    
    # convert into log wages
    return np.log(wages)

# Function to calculate statistics
def calculate_statistics(log_wages):
    employed = np.isnan(log_wages) == False
    median_log_wage = np.nanmedian(log_wages)
    mean_log_wage = np.nanmean(log_wages)
    unemployment_rate = 1 - sum(employed) / len(log_wages)
    #calculate 5, 10, and 25 percentile
    percentile_5 = np.nanpercentile(log_wages, 5)
    percentile_10 = np.nanpercentile(log_wages, 10)
    percentile_15 = np.nanpercentile(log_wages, 15)
    percentile_20 = np.nanpercentile(log_wages, 20)
    percentile_25 = np.nanpercentile(log_wages, 25)

    return median_log_wage, mean_log_wage, unemployment_rate, percentile_5, percentile_10, percentile_15, percentile_20, percentile_25

def generate_weighted_exponential_samples(log_wages, real_m, num_samples, scale=0.5):
    log_m = np.log(real_m)
    # Generate raw exponential samples
    raw_samples = log_m + np.random.exponential(scale=scale, size=3_000)
    # Calculate the KDE of log_wages
    kde = gaussian_kde(log_wages)
    # Assign weights to each point in raw_samples based on the KDE of log_wages
    weights = kde(raw_samples)
    # Normalize the weights
    weights /= np.sum(weights)
    # Resample using the exponential samples and apply the weights
    resampled_with_weights = np.random.choice(raw_samples, size=num_samples, p=weights)
    
    return resampled_with_weights

# calculate_statistics(impose_minimum_wage(generate_log_wages(10000), m, 0.1, 0.2, 0.7)['adjusted_log_wages'])

# Function to impose minimum wage (generalized) with log_wages as input
def impose_minimum_wage(log_wages, real_m, P_o, P_b, P_s, spillover="exponential", spillover_scale=0.1):
    if real_m > 0:    
        log_m = np.log(real_m)
    else:
        log_m = np.log(1e-10)
    # Sanity check
    if P_o + P_b + P_s > 1.0:
        raise ValueError("The sum of probabilities P_o, P_b, and P_s must be less than or equal to 1.")

    original_log_wages = log_wages.copy()
    below_m = log_wages < log_m
    random_probs = np.random.rand(len(log_wages))
    affected = np.zeros(len(log_wages), dtype=bool)

    # Apply the probabilities in a vectorized manner
    unaffected_mask = (below_m) & (random_probs >= P_o + P_b + P_s)
    spillover_mask = (below_m) & (random_probs >= P_o + P_b) & (random_probs < P_o + P_b + P_s)
    bunching_mask = (below_m) & (random_probs >= P_o) & (random_probs < P_o + P_b)
    unemploy_mask = (below_m) & (random_probs < P_o)

    # Check if the sum of all masks equals below_m
    all_masks_sum = unaffected_mask + spillover_mask + unemploy_mask + bunching_mask
    if not np.array_equal(all_masks_sum, below_m):
        raise ValueError("The masks are either overlapping or do not cover all elements of below_m.")

    log_wages[unemploy_mask] = np.nan  # Unemployment
    log_wages[bunching_mask] = log_m  # Bunching to minimum wage
    # Generate spillover samples
    num_samples = np.sum(spillover_mask)
    if spillover == "exponential":
        log_wages[spillover_mask] = log_m + np.random.exponential(scale=spillover_scale, size=num_samples)  # Bunching with spillover
    elif spillover == "proportional_exponential":
        log_wages[spillover_mask] = generate_weighted_exponential_samples(original_log_wages.copy(), real_m, num_samples, scale=spillover_scale)
    else:
        raise ValueError("Invalid spillover method. Choose from 'exponential' or 'proportional_exponential'.")

    # Mark affected rows
    affected = below_m & ~unaffected_mask

    # Create the DataFrame
    df = pd.DataFrame({
        'original_log_wages': original_log_wages,
        'adjusted_log_wages': log_wages,
        'affected': affected,
        'below_m': below_m
    })

    # Sanity check: if affected, adjusted log wages should be >= m or NaN
    assert np.all((df.loc[df['affected'], 'adjusted_log_wages'] >= log_m) | np.isnan(df.loc[df['affected'], 'adjusted_log_wages'])), "Sanity check failed: adjusted log wages are below minimum wage for affected wages."

    return df


def calculate_elasticity(raw_wages, m_before, m_after, P_o=0.2, P_b=0, P_s=0.5):
    
    kargs = (P_o, P_b, P_s)

    df_pre = impose_minimum_wage(raw_wages, m_before, *kargs)
    df_post = impose_minimum_wage(raw_wages, m_after, *kargs)
    # Calculate the number of "employed" and earn below m_after
    n_pre = sum(~np.isnan(df_pre['adjusted_log_wages']))
    n_post = sum(~np.isnan(df_post['adjusted_log_wages']))
    Δe = n_post-n_pre
    n_directly_affected = sum((df_pre['adjusted_log_wages'] < m_after))
    
    Δm = np.exp(m_after) - np.exp(m_before)
    # Calculate the elasticity
    local_elasticity = (Δe/n_directly_affected) / (Δm/np.exp(m_before))
    global_elasticity = (Δe/n_pre) / (Δm/np.exp(m_before))

    return local_elasticity, global_elasticity

# test
# calculate_elasticity(initial_log_wages, 2.2, 2.4)

# Visualization
# Function to plot a single histogram in a panel
def plot_histogram_panel(ax, df, bins, scenario, scenario_params, max_ylim, show_unemployed=False, show_percentile=True):
    original_log_wages = df['original_log_wages']
    adjusted_log_wages = df['adjusted_log_wages']
    affected = df['affected']

    m = scenario_params[0]

    affected_log_wages = adjusted_log_wages[affected]
    unaffected_log_wages = adjusted_log_wages[~affected]

    # Calculate statistics
    median_log_wage, mean_log_wage, unemployment_rate, percentile_5, percentile_10, percentile_15, percentile_20, percentile_25 = calculate_statistics(adjusted_log_wages)
     
    # Get the Spectral color palette
    palette = sns.color_palette("Spectral", 7)

    # Plot original log wages in the background
    ax.hist(original_log_wages[~np.isnan(original_log_wages)], bins=bins, alpha=0.2, label='Original', color=palette[6])

    # Plot unaffected and affected log wages stacked
    ax.hist([unaffected_log_wages[~np.isnan(unaffected_log_wages)], affected_log_wages[~np.isnan(affected_log_wages)]], 
            bins=bins, alpha=0.7, label=['Unaffected', 'Affected'], color=[palette[6], palette[5]], stacked=True)
    
    ax.axvline(np.log(m), color='darkgrey', linestyle='--', label='Minimum wage')
    # add 5, 15, 25 percentile line
    if show_percentile:
        ax.axvline(median_log_wage, color=palette[0], linestyle='-', label='Median log wage')
        ax.axvline(mean_log_wage, color=palette[1], linestyle='-', label='Mean log wage')
        ax.axvline(percentile_5, color=palette[2], linestyle='-', label='5th percentile')
        ax.axvline(percentile_15, color=palette[5], linestyle='-', label='15th percentile')
        ax.axvline(percentile_25, color=palette[6], linestyle='-', label='25th percentile')

        # add value of median to the left of median, and mean to the right of mean
        ax.text(median_log_wage-0.5, max_ylim*0.8, f'{median_log_wage:.2f}', color=palette[0])
        ax.text(mean_log_wage+0.1, max_ylim*0.8, f'{mean_log_wage:.2f}', color=palette[1])

        # add a box on the right bottom of each panel to show the value of 5, 10, 15, 20, 25 percentile
        xplace = 0.3
        ax.text(xplace, max_ylim*0.8, f'$p_{{5}}$: {percentile_5:.2f}', color='black')
        ax.text(xplace, max_ylim*0.73, f'$p_{{10}}$: {percentile_10:.2f}', color='black')
        ax.text(xplace, max_ylim*0.66, f'$p_{{15}}$: {percentile_15:.2f}', color='black')
        ax.text(xplace, max_ylim*0.59, f'$p_{{20}}$: {percentile_20:.2f}', color='black')
        ax.text(xplace, max_ylim*0.52, f'$p_{{25}}$: {percentile_25:.2f}', color='black')

    # add unemployment rate on the middle left of the plot. make a line skip between text and numbers
    if show_unemployed:
        ax.text(0.3, max_ylim*0.3, 'unemp. rate:', color='black')
        ax.text(0.3, max_ylim*0.22, f'{100*unemployment_rate:.2f}%', color='black')
    
    ax.set_title(f'{scenario} ($m$ = {m}, $P_o$={scenario_params[1]}, $P_b$={scenario_params[2]}, $P_s$={scenario_params[3]})')
    ax.set_xlabel('Log Wage')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 7)  # Set xlim to a fixed range
    ax.set_ylim(0, max_ylim)  # Adjust ylim to show the whole histogram
    ax.legend()


# Main function to plot histograms for all scenarios # this function for same m only
def plot_wage_distributions(df_scenarios, scenario_params, max_ylim, show_unemployed):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    bin_width = 0.1
    bins = np.arange(0, 7 + bin_width, bin_width) - 1e-6  # Shifted by a small value to include the rightmost edge
    
    for ax, (scenario, df) in zip(axs.flatten(), df_scenarios.items()):
        plot_histogram_panel(ax, df, bins, scenario, scenario_params[scenario], max_ylim, show_unemployed)
    m = scenario_params['S1'][0]

    # Add common text
    plt.figtext(0.5, 0.04, f'Minimum wage (log): {m}', ha='center', fontsize=12)
    plt.figtext(0.5, 0.01, 'Scenario parameters: $P_o$: unemployed, $P_b$: bunching, $P_s$: spillover', ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# Main function to plot histograms for all scenarios # this function for same m only
def plot_wage_distributions_m(df_scenarios, scenario_params, max_ylim, show_unemployed=True):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    bin_width = 0.1
    bins = np.arange(0, 7 + bin_width, bin_width) - 1e-6  # Shifted by a small value to include the rightmost edge
    
    for ax, (scenario, df) in zip(axs.flatten(), df_scenarios.items()):
        plot_histogram_panel(ax, df, bins, scenario, scenario_params[scenario], max_ylim, show_unemployed)

    # Add common title below the plots
    plt.figtext(0.5, 0.04, 'Wages Distributions with Different Minimum Wages', ha='center', fontsize=14)
    # Add common text
    plt.figtext(0.5, 0.01, 'Scenarios: 20% unemployed + 50% spillover + 30% unaffected', ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
