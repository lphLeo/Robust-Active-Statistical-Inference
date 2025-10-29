import sys
sys.path.insert(1, '../')
import numpy as np
import folktables
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

import scipy
from sklearn.preprocessing import OneHotEncoder
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Rectangle

def get_data(year,features,outcome, randperm=True):
    # Predict income and regress to time to work
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    income_features = acs_data[features].fillna(-1)
    income = acs_data[outcome].fillna(-1)
    employed = np.isin(acs_data['COW'], np.array([1,2,3,4,5,6,7]))
    if randperm:
        shuffler = np.random.permutation(income.shape[0])
        income_features, income, employed = income_features.iloc[shuffler], income.iloc[shuffler], employed[shuffler]
    return income_features, income, employed

def get_data_small(year, features, outcome, m=None, randperm=True):
    # Predict income and regress to time to work
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    income_features = acs_data[features].fillna(-1)
    income = acs_data[outcome].fillna(-1)
    employed = np.isin(acs_data['COW'], np.array([1, 2, 3, 4, 5, 6, 7]))

    if randperm:
        shuffler = np.random.permutation(income.shape[0])
        income_features, income, employed = income_features.iloc[shuffler], income.iloc[shuffler], employed[shuffler]

    if m is not None:
        income_features = income_features.iloc[:m]
        income = income.iloc[:m]
        employed = employed[:m]

    return income_features, income, employed

def transform_features(features, ft, enc=None):
    c_features = features.T[ft == "c"].T.astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse=False)
        enc.fit(c_features)
    c_features = enc.transform(c_features)
    features = scipy.sparse.csc_matrix(np.concatenate([features.T[ft == "q"].T.astype(float), c_features], axis=1))
    return features, enc

def ols(features, outcome):
    ols_coeffs = np.linalg.pinv(features).dot(outcome)
    return ols_coeffs

def make_ess_coverage_plot(dfs, bi_sizes, theta_true, alpha=0.1, n_l=0, n_u=np.inf, filename=None):
    num_bi = len(dfs)
    
    fig, axs = plt.subplots(nrows=2, ncols=num_bi, figsize=(12, 6), 
                            gridspec_kw={'height_ratios': [1, 1]}, sharex='col')
    
    col = ['#721817', '#2B4162', '#FA9F42']
    sns.set_theme(font_scale=1.2, style='white', palette=col, rc={'lines.linewidth': 2})
    all_handles, all_labels = None, None
    processed_dfs_with_ess = []

    for df_orig_bi in dfs:
        df_filtered_pre_ess = df_orig_bi[(df_orig_bi['$n_b$'] > n_l) & (df_orig_bi['$n_b$'] < n_u)]
        if df_filtered_pre_ess.empty:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        ns_local = df_filtered_pre_ess["$n_b$"].unique()
        estimators_local = df_filtered_pre_ess["estimator"].unique()
        
        if len(estimators_local) == 0:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        baseline_estimator_name = estimators_local[0]
        ess_data_list_calc = []

        baseline_nb_points_interp = []
        baseline_mean_widths_interp = []
        for n_b_val_interp in sorted(df_orig_bi["$n_b$"].unique()):
            mean_w_interp = df_orig_bi[(df_orig_bi["estimator"] == baseline_estimator_name) & (df_orig_bi["$n_b$"] == n_b_val_interp)]['interval width'].mean()
            if not np.isnan(mean_w_interp):
                baseline_nb_points_interp.append(n_b_val_interp)
                baseline_mean_widths_interp.append(mean_w_interp)
        
        if not baseline_nb_points_interp or len(baseline_nb_points_interp) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        sorted_indices_interp = np.argsort(baseline_mean_widths_interp)
        interp_x_baseline_widths_unique = np.array(baseline_mean_widths_interp)[sorted_indices_interp]
        interp_y_baseline_nbs_unique = np.array(baseline_nb_points_interp)[sorted_indices_interp]
        
        unique_widths_for_interp, unique_indices_for_interp = np.unique(interp_x_baseline_widths_unique, return_index=True)
        final_interp_x_widths = unique_widths_for_interp
        final_interp_y_nbs = interp_y_baseline_nbs_unique[unique_indices_for_interp]

        if len(final_interp_x_widths) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        for est_name_calc in estimators_local:
            for n_b_val_current_calc in sorted(df_filtered_pre_ess[df_filtered_pre_ess["estimator"] == est_name_calc]["$n_b$"].unique()):
                current_estimator_rows_at_nb_calc = df_filtered_pre_ess[(df_filtered_pre_ess["estimator"] == est_name_calc) & (df_filtered_pre_ess["$n_b$"] == n_b_val_current_calc)]
                if current_estimator_rows_at_nb_calc.empty:
                    continue

                ess_val_calc = np.nan
                if est_name_calc == baseline_estimator_name:
                    ess_val_calc = n_b_val_current_calc
                else:
                    mean_width_current_estimator_calc = current_estimator_rows_at_nb_calc['interval width'].mean()
                    if not np.isnan(mean_width_current_estimator_calc):
                        min_w = np.min(final_interp_x_widths)
                        max_w = np.max(final_interp_x_widths)
                        if min_w <= mean_width_current_estimator_calc <= max_w:
                            ess_val_calc = np.interp(mean_width_current_estimator_calc, final_interp_x_widths, final_interp_y_nbs)
                        else:
                            ess_val_calc = np.nan
                
                for _, row_calc in current_estimator_rows_at_nb_calc.iterrows():
                    ess_data_list_calc.append({
                        '$n_b$': n_b_val_current_calc,
                        'estimator': est_name_calc,
                        'effective_sample_size': ess_val_calc,
                        'coverage': row_calc['coverage'] 
                    })
        
        df_for_this_bi_ess = pd.DataFrame(ess_data_list_calc)
        processed_dfs_with_ess.append(df_for_this_bi_ess)
        
    for i, df_ess_processed in enumerate(processed_dfs_with_ess):
        bi = bi_sizes[i]
        
        current_ax_top = axs[0, i] if num_bi > 1 else axs[0]
        current_ax_bottom = axs[1, i] if num_bi > 1 else axs[1]

        df_ess_valid = df_ess_processed[np.isfinite(df_ess_processed['effective_sample_size'])]

        if df_ess_valid.empty:
            current_ax_top.set_visible(False)
            current_ax_bottom.set_visible(False)
            continue

        valid_nbs_per_estimator = [group["$n_b$"].values for _, group in df_ess_valid.groupby('estimator') if not group.empty]
        if not valid_nbs_per_estimator or any(len(nbs) == 0 for nbs in valid_nbs_per_estimator):
            current_ax_top.set_visible(False)
            current_ax_bottom.set_visible(False)
            continue
        x_min = max([min(nbs) for nbs in valid_nbs_per_estimator])
        x_max = min([max(nbs) for nbs in valid_nbs_per_estimator])
        df_ess_valid_trimmed = df_ess_valid[(df_ess_valid["$n_b$"] >= x_min) & (df_ess_valid["$n_b$"] <= x_max)]
        if df_ess_valid_trimmed.empty:
            current_ax_top.set_visible(False)
            current_ax_bottom.set_visible(False)
            continue
        valid_ns = df_ess_valid_trimmed["$n_b$"].unique()
        min_n_plot = min(valid_ns)
        max_n_plot = max(valid_ns)
        if min_n_plot <= 0 or max_n_plot <=0 or min_n_plot == max_n_plot:
             x_ticks_plot = np.linspace(min_n_plot if min_n_plot > 0 else 1, max_n_plot if max_n_plot >0 else 100, num=4)
        else:
            x_ticks_plot = np.logspace(np.log10(min_n_plot), np.log10(max_n_plot), num=4)
        x_ticks_plot = [int(x) for x in x_ticks_plot]

        min_ess_local = df_ess_valid_trimmed['effective_sample_size'].min()
        max_ess_local = df_ess_valid_trimmed['effective_sample_size'].max()

        if pd.isna(min_ess_local) or pd.isna(max_ess_local) or min_ess_local <= 0 or max_ess_local <= 0 or min_ess_local == max_ess_local:
            ess_y_ticks_local = np.linspace(1, max(1,max_n_plot), 7) 
            min_ess_local_final = 1 
            max_ess_local_final = max(1,max_n_plot)
            if min_ess_local_final == max_ess_local_final : max_ess_local_final = min_ess_local_final * 10 
        else:
            ess_y_ticks_local = np.logspace(np.log10(min_ess_local), np.log10(max_ess_local), num=7)
            min_ess_local_final = min_ess_local
            max_ess_local_final = max_ess_local

        # Effective Sample Size
        sns.lineplot(ax=current_ax_top, data=df_ess_valid_trimmed, x='$n_b$', y='effective_sample_size', 
                     hue='estimator', alpha=0.8)
        current_ax_top.set(xscale='log', yscale='log')
        current_ax_top.set_yticks(ess_y_ticks_local)
        current_ax_top.xaxis.set_minor_formatter(NullFormatter())
        current_ax_top.yaxis.set_minor_formatter(NullFormatter())
        current_ax_top.get_xaxis().set_major_formatter(ScalarFormatter())
        current_ax_top.get_yaxis().set_major_formatter(ScalarFormatter())
        
        if i == 0:
            current_ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            current_ax_top.set_ylabel("Effective sample size", fontsize=20)
        else:
            current_ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            current_ax_top.set_ylabel("")
        
        current_ax_top.grid(True, alpha=0.3)
        current_ax_top.add_patch(
            Rectangle( (0, 1), 1, 0.16, transform=current_ax_top.transAxes, facecolor='lightgrey',
                       edgecolor='black', linewidth=0.8, zorder=2, clip_on=False))
        current_ax_top.text(0.5, 1.05 + (0.13 - 0.09)/2, f"Burn-in size = {bi}", ha='center', va='center',
                           fontsize=13, color='black', transform=current_ax_top.transAxes, zorder=3)
        
        current_ax_top.set_ylim(min_ess_local_final , max_ess_local_final )
        current_ax_top.set_xlim([min_n_plot, max_n_plot])
        current_ax_top.get_legend().remove()
        
        # Coverage
        sns.lineplot(ax=current_ax_bottom, data=df_ess_valid_trimmed, x='$n_b$', y='coverage', 
                     hue='estimator', alpha=0.8, errorbar=None)
        current_ax_bottom.axhline(1-alpha, color="#888888", linestyle='dashed', zorder=1, alpha=0.8)
        current_ax_bottom.set_ylim([0.6, 1.0])
        current_ax_bottom.set_xticks(x_ticks_plot if i == 0 or num_bi == 1 else x_ticks_plot[1:])
        
        current_ax_bottom.set_xlim([min_n_plot, max_n_plot])
        current_ax_bottom.grid(True, alpha=0.3)
        current_ax_bottom.set_xlabel("")
        
        if i == 0:
            current_ax_bottom.set_ylabel("Coverage", fontsize=20)
        else:
            current_ax_bottom.set_ylabel("")
            current_ax_bottom.set_yticklabels([])
        
        if i == num_bi - 1 or num_bi == 1:
            handles_leg, labels_leg = current_ax_bottom.get_legend_handles_labels()
            if handles_leg:
                 all_handles, all_labels = handles_leg, labels_leg
        
        current_ax_bottom.get_legend().remove()
        current_ax_top.set_xlabel("")
        current_ax_top.set_yticklabels(current_ax_top.get_yticklabels(), fontsize=13)
        current_ax_bottom.set_xticklabels(current_ax_bottom.get_xticklabels(), fontsize=13)
        current_ax_bottom.set_yticklabels(current_ax_bottom.get_yticklabels(), fontsize=13)

    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0 + (0.13*0.2)),
                   ncol=len(all_handles) if all_handles else 3, fancybox=True, shadow=False, fontsize=20)
    fig.supxlabel("$n_b$", fontsize=20)
    
    if num_bi > 1:
        fig.align_ylabels(axs[:, 0])
    elif num_bi == 1 and axs is not None:
         fig.align_ylabels(axs)

    plt.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.35, hspace=0.2)
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        
    return fig

def make_ess_coverage_plot_custom_title(dfs, title, theta_true, alpha=0.1, n_l=0, n_u=np.inf, filename=None):
    num_bi = len(dfs)
    
    fig, axs = plt.subplots(nrows=2, ncols=num_bi, figsize=(12, 6), 
                            gridspec_kw={'height_ratios': [1, 1]}, sharex='col')
    
    col = ['#721817', '#2B4162', '#FA9F42']
    # col = ['#49BEAA', '#456990', '#EF767A']
    sns.set_theme(font_scale=1.2, style='white', palette=col, rc={'lines.linewidth': 2})
    
    all_handles, all_labels = None, None
    
    processed_dfs_with_ess = []

    for df_orig_bi in dfs:
        df_filtered_pre_ess = df_orig_bi[(df_orig_bi['$n_b$'] > n_l) & (df_orig_bi['$n_b$'] < n_u)]
        if df_filtered_pre_ess.empty:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        ns_local = df_filtered_pre_ess["$n_b$"].unique()
        estimators_local = df_filtered_pre_ess["estimator"].unique()
        
        if len(estimators_local) == 0:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        baseline_estimator_name = estimators_local[0]
        ess_data_list_calc = []

        baseline_nb_points_interp = []
        baseline_mean_widths_interp = []
        for n_b_val_interp in sorted(df_orig_bi["$n_b$"].unique()):
            mean_w_interp = df_orig_bi[(df_orig_bi["estimator"] == baseline_estimator_name) & (df_orig_bi["$n_b$"] == n_b_val_interp)]['interval width'].mean()
            if not np.isnan(mean_w_interp):
                baseline_nb_points_interp.append(n_b_val_interp)
                baseline_mean_widths_interp.append(mean_w_interp)
        
        if not baseline_nb_points_interp or len(baseline_nb_points_interp) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        sorted_indices_interp = np.argsort(baseline_mean_widths_interp)
        interp_x_baseline_widths_unique = np.array(baseline_mean_widths_interp)[sorted_indices_interp]
        interp_y_baseline_nbs_unique = np.array(baseline_nb_points_interp)[sorted_indices_interp]
        
        unique_widths_for_interp, unique_indices_for_interp = np.unique(interp_x_baseline_widths_unique, return_index=True)
        final_interp_x_widths = unique_widths_for_interp
        final_interp_y_nbs = interp_y_baseline_nbs_unique[unique_indices_for_interp]

        if len(final_interp_x_widths) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue

        for est_name_calc in estimators_local:
            for n_b_val_current_calc in sorted(df_filtered_pre_ess[df_filtered_pre_ess["estimator"] == est_name_calc]["$n_b$"].unique()):
                current_estimator_rows_at_nb_calc = df_filtered_pre_ess[(df_filtered_pre_ess["estimator"] == est_name_calc) & (df_filtered_pre_ess["$n_b$"] == n_b_val_current_calc)]
                if current_estimator_rows_at_nb_calc.empty:
                    continue

                ess_val_calc = np.nan
                if est_name_calc == baseline_estimator_name:
                    ess_val_calc = n_b_val_current_calc
                else:
                    mean_width_current_estimator_calc = current_estimator_rows_at_nb_calc['interval width'].mean()
                    if not np.isnan(mean_width_current_estimator_calc):
                        min_w = np.min(final_interp_x_widths)
                        max_w = np.max(final_interp_x_widths)
                        if min_w <= mean_width_current_estimator_calc <= max_w:
                            ess_val_calc = np.interp(mean_width_current_estimator_calc, final_interp_x_widths, final_interp_y_nbs)
                        else:
                            ess_val_calc = np.nan
                
                for _, row_calc in current_estimator_rows_at_nb_calc.iterrows():
                    ess_data_list_calc.append({
                        '$n_b$': n_b_val_current_calc,
                        'estimator': est_name_calc,
                        'effective_sample_size': ess_val_calc,
                        'coverage': row_calc['coverage'] 
                    })
        
        df_for_this_bi_ess = pd.DataFrame(ess_data_list_calc)
        processed_dfs_with_ess.append(df_for_this_bi_ess)
        
    for i, df_ess_processed in enumerate(processed_dfs_with_ess):
        
        current_ax_top = axs[0, i] if num_bi > 1 else axs[0]
        current_ax_bottom = axs[1, i] if num_bi > 1 else axs[1]

        df_ess_valid = df_ess_processed[np.isfinite(df_ess_processed['effective_sample_size'])]

        if df_ess_valid.empty:
            current_ax_top.set_visible(False)
            current_ax_bottom.set_visible(False)
            continue

        valid_nbs_per_estimator = [group["$n_b$"].values for _, group in df_ess_valid.groupby('estimator') if not group.empty]
        if not valid_nbs_per_estimator or any(len(nbs) == 0 for nbs in valid_nbs_per_estimator):
            current_ax_top.set_visible(False)
            current_ax_bottom.set_visible(False)
            continue
        x_min = max([min(nbs) for nbs in valid_nbs_per_estimator])
        x_max = min([max(nbs) for nbs in valid_nbs_per_estimator])

        df_ess_valid_trimmed = df_ess_valid[(df_ess_valid["$n_b$"] >= x_min) & (df_ess_valid["$n_b$"] <= x_max)]
        if df_ess_valid_trimmed.empty:
            current_ax_top.set_visible(False)
            current_ax_bottom.set_visible(False)
            continue
        valid_ns = df_ess_valid_trimmed["$n_b$"].unique()
        min_n_plot = min(valid_ns)
        max_n_plot = max(valid_ns)
        if min_n_plot <= 0 or max_n_plot <=0 or min_n_plot == max_n_plot:
             x_ticks_plot = np.linspace(min_n_plot if min_n_plot > 0 else 1, max_n_plot if max_n_plot >0 else 100, num=4)
        else:
            x_ticks_plot = np.logspace(np.log10(min_n_plot), np.log10(max_n_plot), num=4)
        x_ticks_plot = [int(x) for x in x_ticks_plot]

        min_ess_local = df_ess_valid_trimmed['effective_sample_size'].min()
        max_ess_local = df_ess_valid_trimmed['effective_sample_size'].max()

        if pd.isna(min_ess_local) or pd.isna(max_ess_local) or min_ess_local <= 0 or max_ess_local <= 0 or min_ess_local == max_ess_local:
            ess_y_ticks_local = np.linspace(1, max(1,max_n_plot), 7) 
            min_ess_local_final = 1 
            max_ess_local_final = max(1,max_n_plot)
            if min_ess_local_final == max_ess_local_final : max_ess_local_final = min_ess_local_final * 10
        else:
            ess_y_ticks_local = np.logspace(np.log10(min_ess_local), np.log10(max_ess_local), num=7)
            min_ess_local_final = min_ess_local
            max_ess_local_final = max_ess_local

        # Effective Sample Size
        sns.lineplot(ax=current_ax_top, data=df_ess_valid_trimmed, x='$n_b$', y='effective_sample_size', 
                     hue='estimator', alpha=0.8)
        current_ax_top.set(xscale='log', yscale='log')
        current_ax_top.set_yticks(ess_y_ticks_local)
        current_ax_top.xaxis.set_minor_formatter(NullFormatter())
        current_ax_top.yaxis.set_minor_formatter(NullFormatter())
        current_ax_top.get_xaxis().set_major_formatter(ScalarFormatter())
        current_ax_top.get_yaxis().set_major_formatter(ScalarFormatter())
        current_ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        if i == 0:
            current_ax_top.set_ylabel("Effective sample size", fontsize=20)
        else:
            current_ax_top.set_ylabel("")
        
        current_ax_top.grid(True, alpha=0.3)
        current_ax_top.add_patch(
            Rectangle( (0, 1), 1, 0.16, transform=current_ax_top.transAxes, facecolor='lightgrey',
                       edgecolor='black', linewidth=0.8, zorder=2, clip_on=False))
        current_ax_top.text(0.5, 1.05 + (0.13 - 0.09)/2, title[i], ha='center', va='center',
                           fontsize=20, color='black', transform=current_ax_top.transAxes, zorder=3)
        
        current_ax_top.set_ylim(min_ess_local_final , max_ess_local_final )
        current_ax_top.set_xlim([min_n_plot, max_n_plot])
        current_ax_top.get_legend().remove()
        
        # Coverage
        sns.lineplot(ax=current_ax_bottom, data=df_ess_valid_trimmed, x='$n_b$', y='coverage', 
                     hue='estimator', alpha=0.8, errorbar=None)
        current_ax_bottom.axhline(1-alpha, color="#888888", linestyle='dashed', zorder=1, alpha=0.8)
        current_ax_bottom.set_ylim([0.6, 1.0])
        current_ax_bottom.set_xticks(x_ticks_plot if i == 0 or num_bi == 1 else x_ticks_plot[1:])
        
        current_ax_bottom.set_xlim([min_n_plot, max_n_plot])
        current_ax_bottom.grid(True, alpha=0.3)
        current_ax_bottom.set_xlabel("")
        
        if i == 0:
            current_ax_bottom.set_ylabel("Coverage", fontsize=20)
        else:
            current_ax_bottom.set_ylabel("")
            current_ax_bottom.set_yticklabels([])
        
        if i == num_bi - 1 or num_bi == 1:
            handles_leg, labels_leg = current_ax_bottom.get_legend_handles_labels()
            if handles_leg:
                 all_handles, all_labels = handles_leg, labels_leg
        
        current_ax_bottom.get_legend().remove()
        current_ax_top.set_xlabel("")
        current_ax_top.set_yticklabels(current_ax_top.get_yticklabels(), fontsize=16)
        current_ax_bottom.set_xticklabels(current_ax_bottom.get_xticklabels(), fontsize=16)
        current_ax_bottom.set_yticklabels(current_ax_bottom.get_yticklabels(), fontsize=16)

    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0 + (0.13*0.2)), fontsize=20,
                   ncol=len(all_handles) if all_handles else 3, fancybox=True, shadow=False)
    fig.supxlabel("$n_b$", fontsize=20)
    
    if num_bi > 1:
        fig.align_ylabels(axs[:, 0])
    elif num_bi == 1 and axs is not None:
         fig.align_ylabels(axs) 

    plt.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.3, hspace=0.2)
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        
    return fig

def make_ess_plot(dfs, bi_sizes, theta_true, alpha=0.1, n_l=0, n_u=np.inf, filename=None):
    num_bi = len(dfs)
    fig, axs = plt.subplots(nrows=1, ncols=num_bi, figsize=(12, 4), sharex=False)
    col = ['#721817', '#2B4162', '#FA9F42']
    sns.set_theme(font_scale=1.2, style='white', palette=col, rc={'lines.linewidth': 2})
    all_handles, all_labels = None, None
    processed_dfs_with_ess = []

    for df_orig_bi in dfs:
        df_filtered_pre_ess = df_orig_bi[(df_orig_bi['$n_b$'] > n_l) & (df_orig_bi['$n_b$'] < n_u)]
        if df_filtered_pre_ess.empty:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        ns_local = df_filtered_pre_ess["$n_b$"].unique()
        estimators_local = df_filtered_pre_ess["estimator"].unique()
        if len(estimators_local) == 0:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        baseline_estimator_name = estimators_local[0]
        ess_data_list_calc = []
        baseline_nb_points_interp = []
        baseline_mean_widths_interp = []
        for n_b_val_interp in sorted(df_orig_bi["$n_b$"].unique()):
            mean_w_interp = df_orig_bi[(df_orig_bi["estimator"] == baseline_estimator_name) & (df_orig_bi["$n_b$"] == n_b_val_interp)]['interval width'].mean()
            if not np.isnan(mean_w_interp):
                baseline_nb_points_interp.append(n_b_val_interp)
                baseline_mean_widths_interp.append(mean_w_interp)
        if not baseline_nb_points_interp or len(baseline_nb_points_interp) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        sorted_indices_interp = np.argsort(baseline_mean_widths_interp)
        interp_x_baseline_widths_unique = np.array(baseline_mean_widths_interp)[sorted_indices_interp]
        interp_y_baseline_nbs_unique = np.array(baseline_nb_points_interp)[sorted_indices_interp]
        unique_widths_for_interp, unique_indices_for_interp = np.unique(interp_x_baseline_widths_unique, return_index=True)
        final_interp_x_widths = unique_widths_for_interp
        final_interp_y_nbs = interp_y_baseline_nbs_unique[unique_indices_for_interp]
        if len(final_interp_x_widths) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        for est_name_calc in estimators_local:
            for n_b_val_current_calc in sorted(df_filtered_pre_ess[df_filtered_pre_ess["estimator"] == est_name_calc]["$n_b$"].unique()):
                current_estimator_rows_at_nb_calc = df_filtered_pre_ess[(df_filtered_pre_ess["estimator"] == est_name_calc) & (df_filtered_pre_ess["$n_b$"] == n_b_val_current_calc)]
                if current_estimator_rows_at_nb_calc.empty:
                    continue
                ess_val_calc = np.nan
                if est_name_calc == baseline_estimator_name:
                    ess_val_calc = n_b_val_current_calc
                else:
                    mean_width_current_estimator_calc = current_estimator_rows_at_nb_calc['interval width'].mean()
                    if not np.isnan(mean_width_current_estimator_calc):
                        min_w = np.min(final_interp_x_widths)
                        max_w = np.max(final_interp_x_widths)
                        if min_w <= mean_width_current_estimator_calc <= max_w:
                            ess_val_calc = np.interp(mean_width_current_estimator_calc, final_interp_x_widths, final_interp_y_nbs)
                        else:
                            ess_val_calc = np.nan
                
                for _, row_calc in current_estimator_rows_at_nb_calc.iterrows():
                    ess_data_list_calc.append({
                        '$n_b$': n_b_val_current_calc,
                        'estimator': est_name_calc,
                        'effective_sample_size': ess_val_calc
                    })
        df_for_this_bi_ess = pd.DataFrame(ess_data_list_calc)
        processed_dfs_with_ess.append(df_for_this_bi_ess)

    for i, df_ess_processed in enumerate(processed_dfs_with_ess):
        bi = bi_sizes[i]
        current_ax = axs[i] if num_bi > 1 else axs
        df_ess_valid = df_ess_processed[np.isfinite(df_ess_processed['effective_sample_size'])]
        if df_ess_valid.empty:
            current_ax.set_visible(False)
            continue
        valid_nbs_per_estimator = [group["$n_b$"].values for _, group in df_ess_valid.groupby('estimator') if not group.empty]
        if not valid_nbs_per_estimator or any(len(nbs) == 0 for nbs in valid_nbs_per_estimator):
            current_ax.set_visible(False)
            continue
        x_min = max([min(nbs) for nbs in valid_nbs_per_estimator])
        x_max = min([max(nbs) for nbs in valid_nbs_per_estimator])
        df_ess_valid_trimmed = df_ess_valid[(df_ess_valid["$n_b$"] >= x_min) & (df_ess_valid["$n_b$"] <= x_max)]
        if df_ess_valid_trimmed.empty:
            current_ax.set_visible(False)
            continue
        valid_ns = df_ess_valid_trimmed["$n_b$"].unique()
        min_n_plot = min(valid_ns)
        max_n_plot = max(valid_ns)
        if min_n_plot <= 0 or max_n_plot <=0 or min_n_plot == max_n_plot:
             x_ticks_plot = np.linspace(min_n_plot if min_n_plot > 0 else 1, max_n_plot if max_n_plot >0 else 100, num=4)
        else:
            x_ticks_plot = np.logspace(np.log10(min_n_plot), np.log10(max_n_plot), num=4)
        x_ticks_plot = [int(x) for x in x_ticks_plot]
        min_ess_local = df_ess_valid_trimmed['effective_sample_size'].min()
        max_ess_local = df_ess_valid_trimmed['effective_sample_size'].max()
        if pd.isna(min_ess_local) or pd.isna(max_ess_local) or min_ess_local <= 0 or max_ess_local <= 0 or min_ess_local == max_ess_local:
            ess_y_ticks_local = np.linspace(1, max(1,max_n_plot), 7)
            min_ess_local_final = 1 
            max_ess_local_final = max(1,max_n_plot)
            if min_ess_local_final == max_ess_local_final : max_ess_local_final = min_ess_local_final * 10
        else:
            ess_y_ticks_local = np.logspace(np.log10(min_ess_local), np.log10(max_ess_local), num=7)
            min_ess_local_final = min_ess_local
            max_ess_local_final = max_ess_local
        sns.lineplot(ax=current_ax, data=df_ess_valid_trimmed, x='$n_b$', y='effective_sample_size', 
                     hue='estimator', alpha=0.8)
        current_ax.set(xscale='log', yscale='log')
        current_ax.set_yticks(ess_y_ticks_local)
        current_ax.xaxis.set_minor_formatter(NullFormatter())
        current_ax.yaxis.set_minor_formatter(NullFormatter())
        current_ax.get_xaxis().set_major_formatter(ScalarFormatter())
        current_ax.get_yaxis().set_major_formatter(ScalarFormatter())
        current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if i == 0:
            current_ax.set_ylabel("Effective sample size", fontsize=20)
        else:
            current_ax.set_ylabel("")
        current_ax.grid(True, alpha=0.3)
        current_ax.add_patch(
            Rectangle( (0, 1), 1, 0.16, transform=current_ax.transAxes, facecolor='lightgrey',
                       edgecolor='black', linewidth=0.8, zorder=2, clip_on=False))
        current_ax.text(0.5, 1.05 + (0.13 - 0.09)/2, f"Burn-in size = {bi}", ha='center', va='center',
                           fontsize=14, color='black', transform=current_ax.transAxes, zorder=3)
        current_ax.set_ylim(min_ess_local_final , max_ess_local_final )
        current_ax.set_xlim([min_n_plot, max_n_plot])
        current_ax.set_xticks(x_ticks_plot[1:])
        current_ax.set_xlabel('')
        current_ax.set_xticklabels(current_ax.get_xticklabels(), fontsize=12)
        current_ax.set_yticklabels(current_ax.get_yticklabels(), fontsize=12)
        current_ax.get_legend().remove()
        if i == num_bi - 1 or num_bi == 1:
            handles_leg, labels_leg = current_ax.get_legend_handles_labels()
            if handles_leg:
                 all_handles, all_labels = handles_leg, labels_leg
    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0 + (0.13*0.2)),
                   ncol=len(all_handles) if all_handles else 3, fancybox=True, shadow=False, fontsize=20)
    fig.supxlabel("$n_b$", fontsize=20)
    if num_bi > 1:
        fig.align_ylabels(axs)
    plt.tight_layout()
    fig.subplots_adjust(top=0.7, bottom=0.22, wspace=0.35, hspace=0.25)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    return fig

def make_ess_coverage_plot_intro(df, estimand_title, filename, theta_true, alpha = 0.1, n_l = 0, n_u = np.inf, num_trials = 100, n_example_ind = 0, finetuning=False, more_precision=False, less_precision=False):
    ns = df["$n_b$"].unique()
    estimators = df["estimator"].unique()
    widths = np.zeros((len(estimators), len(ns)))

    for i in range(len(estimators)):
        for j in range(len(ns)):
            widths[i,j] = df[(df.estimator == estimators[i]) & (df["$n_b$"] == ns[j])]['interval width'].mean()

    baseline_widths = widths[0, :]
    ess = np.full_like(widths, np.nan)
    ess[0, :] = ns 
    for i in range(1, len(estimators)):
        for j in range(len(ns)):
            w = widths[i, j]
            if np.min(baseline_widths) <= w <= np.max(baseline_widths):
                ess[i, j] = np.interp(w, baseline_widths[::-1], ns[::-1])  
    plot_data = []
    for i, est in enumerate(estimators):
        for j, n_b in enumerate(ns):
            if not np.isnan(ess[i, j]):
                plot_data.append({
                    '$n_b$': n_b,
                    'estimator': est,
                    'effective_sample_size': ess[i, j]
                })
    df_ess = pd.DataFrame(plot_data)

    valid_nbs_per_estimator = [df_ess[df_ess['estimator'] == est]['$n_b$'].values for est in estimators]
    x_min = max([min(nbs) for nbs in valid_nbs_per_estimator if len(nbs) > 0])
    x_max = min([max(nbs) for nbs in valid_nbs_per_estimator if len(nbs) > 0])
    df_ess = df_ess[(df_ess['$n_b$'] >= x_min) & (df_ess['$n_b$'] <= x_max)]

    col = [sns.color_palette("pastel")[1], sns.color_palette("pastel")[2], sns.color_palette("pastel")[0]]
    col[2] = 'pink'
    col = ['#721817', '#2B4162', '#FA9F42']
    sns.set_theme(font_scale=1.2, style='white', palette=col, rc={'lines.linewidth': 2})
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4.5))

    # ESS plot
    sns.lineplot(ax=axs[0], data=df_ess, x='$n_b$', y='effective_sample_size', hue='estimator', alpha=0.8)
    axs[0].set(xscale='log', yscale='log')
    axs[0].xaxis.set_minor_formatter(NullFormatter())
    axs[0].yaxis.set_minor_formatter(NullFormatter())
    axs[0].get_xaxis().set_major_formatter(ScalarFormatter())
    axs[0].get_yaxis().set_major_formatter(ScalarFormatter())
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[0].set_ylabel("Effective sample size", fontsize=20, labelpad=10)
    axs[0].set_xlabel('$n_b$', fontsize=20)

    min_n = df_ess['$n_b$'].min()
    max_n = df_ess['$n_b$'].max()
    if min_n > 0 and max_n > 0 and min_n != max_n:
        x_ticks = np.logspace(np.log10(min_n), np.log10(max_n), num=4)
        x_ticks = [int(x) for x in x_ticks]
        axs[0].set_xticks(x_ticks)
        axs[1].set_xticks(x_ticks)
        axs[0].set_xlim([min_n, max_n])
        axs[1].set_xlim([min_n, max_n])
    else:
        axs[0].set_xticklabels(axs[0].get_xticks(), fontsize=16)
        axs[1].set_xticklabels(axs[1].get_xticks(), fontsize=16)

    min_ess = df_ess['effective_sample_size'].min()
    max_ess = df_ess['effective_sample_size'].max()
    if min_ess > 0 and max_ess > 0 and min_ess != max_ess:
        y_ticks = np.logspace(np.log10(min_ess), np.log10(max_ess), num=7)
        y_ticks = [int(y) for y in y_ticks]
        axs[0].set_yticks(y_ticks)
        axs[0].set_yticklabels(y_ticks, fontsize=16)
    else:
        axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=16)

    axs[0].grid(True)
    axs[0].get_legend().remove()

    # Coverage plot
    sns.lineplot(ax=axs[1], data=df[(df['$n_b$'] > n_l) & (df['$n_b$'] < n_u)], x='$n_b$', y='coverage', hue='estimator', alpha=0.8, errorbar=None)
    axs[1].axhline(1-alpha, color="#888888", linestyle='dashed', zorder=1, alpha=0.8)
    axs[1].set_ylim([0.6, 1.0])  
    axs[1].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])  
    
    x_ticks_coverage = np.linspace(np.min(ns[ns>n_l]), np.max(ns[ns<n_u]), 5)
    x_ticks_coverage = [int(x) for x in x_ticks_coverage]
    axs[1].set_xticks(x_ticks_coverage)
    axs[1].set_xlim([np.min(ns[ns>n_l]), np.max(ns[ns<n_u])])
    
    axs[1].set_xlabel('$n_b$', fontsize=20)
    axs[1].set_ylabel("Coverage", fontsize=20, labelpad=10)
    axs[1].grid(True)
    axs[1].get_legend().remove()
    handles, labels = axs[1].get_legend_handles_labels()
    
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=16)
    axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=16)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=16)
    axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.25, hspace=0.25)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0 + (0.13*0.2)),
                   ncol=len(handles) if handles else 3, fancybox=True, shadow=False, fontsize=20)
    fig.supxlabel("$n_b$", fontsize=20)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def make_ess_plot_custom_title(dfs, title, theta_true, alpha=0.1, n_l=0, n_u=np.inf, filename=None):
    num_bi = len(dfs)
    fig, axs = plt.subplots(nrows=1, ncols=num_bi, figsize=(12, 4), sharex=False)
    col = ['#721817', '#2B4162', '#FA9F42']
    # col = ['#49BEAA', '#456990', '#EF767A']
    sns.set_theme(font_scale=1.2, style='white', palette=col, rc={'lines.linewidth': 2})
    all_handles, all_labels = None, None
    processed_dfs_with_ess = []

    for df_orig_bi in dfs:
        df_filtered_pre_ess = df_orig_bi[(df_orig_bi['$n_b$'] > n_l) & (df_orig_bi['$n_b$'] < n_u)]
        if df_filtered_pre_ess.empty:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        ns_local = df_filtered_pre_ess["$n_b$"].unique()
        estimators_local = df_filtered_pre_ess["estimator"].unique()
        if len(estimators_local) == 0:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        baseline_estimator_name = estimators_local[0]
        ess_data_list_calc = []
        baseline_nb_points_interp = []
        baseline_mean_widths_interp = []
        for n_b_val_interp in sorted(df_orig_bi["$n_b$"].unique()):
            mean_w_interp = df_orig_bi[(df_orig_bi["estimator"] == baseline_estimator_name) & (df_orig_bi["$n_b$"] == n_b_val_interp)]['interval width'].mean()
            if not np.isnan(mean_w_interp):
                baseline_nb_points_interp.append(n_b_val_interp)
                baseline_mean_widths_interp.append(mean_w_interp)
        if not baseline_nb_points_interp or len(baseline_nb_points_interp) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        sorted_indices_interp = np.argsort(baseline_mean_widths_interp)
        interp_x_baseline_widths_unique = np.array(baseline_mean_widths_interp)[sorted_indices_interp]
        interp_y_baseline_nbs_unique = np.array(baseline_nb_points_interp)[sorted_indices_interp]
        unique_widths_for_interp, unique_indices_for_interp = np.unique(interp_x_baseline_widths_unique, return_index=True)
        final_interp_x_widths = unique_widths_for_interp
        final_interp_y_nbs = interp_y_baseline_nbs_unique[unique_indices_for_interp]
        if len(final_interp_x_widths) < 2:
            processed_dfs_with_ess.append(pd.DataFrame())
            continue
        for est_name_calc in estimators_local:
            for n_b_val_current_calc in sorted(df_filtered_pre_ess[df_filtered_pre_ess["estimator"] == est_name_calc]["$n_b$"].unique()):
                current_estimator_rows_at_nb_calc = df_filtered_pre_ess[(df_filtered_pre_ess["estimator"] == est_name_calc) & (df_filtered_pre_ess["$n_b$"] == n_b_val_current_calc)]
                if current_estimator_rows_at_nb_calc.empty:
                    continue
                ess_val_calc = np.nan
                if est_name_calc == baseline_estimator_name:
                    ess_val_calc = n_b_val_current_calc
                else:
                    mean_width_current_estimator_calc = current_estimator_rows_at_nb_calc['interval width'].mean()
                    if not np.isnan(mean_width_current_estimator_calc):
                        min_w = np.min(final_interp_x_widths)
                        max_w = np.max(final_interp_x_widths)
                        if min_w <= mean_width_current_estimator_calc <= max_w:
                            ess_val_calc = np.interp(mean_width_current_estimator_calc, final_interp_x_widths, final_interp_y_nbs)
                        else:
                            ess_val_calc = np.nan
                
                for _, row_calc in current_estimator_rows_at_nb_calc.iterrows():
                    ess_data_list_calc.append({
                        '$n_b$': n_b_val_current_calc,
                        'estimator': est_name_calc,
                        'effective_sample_size': ess_val_calc
                    })
        df_for_this_bi_ess = pd.DataFrame(ess_data_list_calc)
        processed_dfs_with_ess.append(df_for_this_bi_ess)

    for i, df_ess_processed in enumerate(processed_dfs_with_ess):
        current_ax = axs[i] if num_bi > 1 else axs
        df_ess_valid = df_ess_processed[np.isfinite(df_ess_processed['effective_sample_size'])]
        if df_ess_valid.empty:
            current_ax.set_visible(False)
            continue
        valid_nbs_per_estimator = [group["$n_b$"].values for _, group in df_ess_valid.groupby('estimator') if not group.empty]
        if not valid_nbs_per_estimator or any(len(nbs) == 0 for nbs in valid_nbs_per_estimator):
            current_ax.set_visible(False)
            continue
        x_min = max([min(nbs) for nbs in valid_nbs_per_estimator])
        x_max = min([max(nbs) for nbs in valid_nbs_per_estimator])
        df_ess_valid_trimmed = df_ess_valid[(df_ess_valid["$n_b$"] >= x_min) & (df_ess_valid["$n_b$"] <= x_max)]
        if df_ess_valid_trimmed.empty:
            current_ax.set_visible(False)
            continue
        valid_ns = df_ess_valid_trimmed["$n_b$"].unique()
        min_n_plot = min(valid_ns)
        max_n_plot = max(valid_ns)
        if min_n_plot <= 0 or max_n_plot <=0 or min_n_plot == max_n_plot:
             x_ticks_plot = np.linspace(min_n_plot if min_n_plot > 0 else 1, max_n_plot if max_n_plot >0 else 100, num=4)
        else:
            x_ticks_plot = np.logspace(np.log10(min_n_plot), np.log10(max_n_plot), num=4)
        x_ticks_plot = [int(x) for x in x_ticks_plot]
        min_ess_local = df_ess_valid_trimmed['effective_sample_size'].min()
        max_ess_local = df_ess_valid_trimmed['effective_sample_size'].max()
        if pd.isna(min_ess_local) or pd.isna(max_ess_local) or min_ess_local <= 0 or max_ess_local <= 0 or min_ess_local == max_ess_local:
            ess_y_ticks_local = np.linspace(1, max(1,max_n_plot), 7)
            min_ess_local_final = 1 
            max_ess_local_final = max(1,max_n_plot)
            if min_ess_local_final == max_ess_local_final : max_ess_local_final = min_ess_local_final * 10
        else:
            ess_y_ticks_local = np.logspace(np.log10(min_ess_local), np.log10(max_ess_local), num=7)
            min_ess_local_final = min_ess_local
            max_ess_local_final = max_ess_local
        sns.lineplot(ax=current_ax, data=df_ess_valid_trimmed, x='$n_b$', y='effective_sample_size', 
                     hue='estimator', alpha=0.8)
        current_ax.set(xscale='log', yscale='log')
        current_ax.set_yticks(ess_y_ticks_local)
        current_ax.xaxis.set_minor_formatter(NullFormatter())
        current_ax.yaxis.set_minor_formatter(NullFormatter())
        current_ax.get_xaxis().set_major_formatter(ScalarFormatter())
        current_ax.get_yaxis().set_major_formatter(ScalarFormatter())
        current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if i == 0:
            current_ax.set_ylabel("Effective sample size", fontsize=20, labelpad=10)
        else:
            current_ax.set_ylabel("")
        current_ax.grid(True, alpha=0.3)
        current_ax.add_patch(
            Rectangle( (0, 1), 1, 0.16, transform=current_ax.transAxes, facecolor='lightgrey',
                       edgecolor='black', linewidth=0.8, zorder=2, clip_on=False))
        current_ax.text(0.5, 1.05 + (0.13 - 0.09)/2, title[i], ha='center', va='center',
                           fontsize=20, color='black', transform=current_ax.transAxes, zorder=3)
        current_ax.set_ylim(min_ess_local_final , max_ess_local_final )
        current_ax.set_xlim([min_n_plot, max_n_plot])
        current_ax.set_xticks(x_ticks_plot[1:])
        current_ax.set_xticklabels(current_ax.get_xticklabels(), fontsize=16)
        current_ax.set_yticklabels(current_ax.get_yticklabels(), fontsize=16)
        current_ax.set_xlabel('')
        current_ax.get_legend().remove()
        if i == num_bi - 1 or num_bi == 1:
            handles_leg, labels_leg = current_ax.get_legend_handles_labels()
            if handles_leg:
                 all_handles, all_labels = handles_leg, labels_leg
    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0 + (0.13*0.2)),
                   ncol=len(all_handles) if all_handles else 3, fancybox=True, shadow=False, fontsize=20)
    fig.supxlabel("$n_b$", fontsize=20)
    if num_bi > 1:
        fig.align_ylabels(axs)
    plt.tight_layout()
    fig.subplots_adjust(top=0.7, bottom=0.22, wspace=0.28, hspace=0.25)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    return fig


def safe_log1pexp(x):
    """
    Compute log(1 + exp(x)) in a numerically stable way.
    """
    idxs = x > 10
    out = np.empty_like(x)
    out[idxs] = x[idxs]
    out[~idxs] = np.log1p(np.exp(x[~idxs]))
    return out

def safe_expit(x):
    """Computes the sigmoid function in a numerically stable way."""
    return np.exp(-np.logaddexp(0, -x))

def opt_mean_tuning(Y, Yhat, weights, sampling_ratio):
    return np.clip(np.mean(Y*Yhat*weights*sampling_ratio)/np.mean(Yhat**2*sampling_ratio), 0, 1)

def logistic(X, Y):
    regression = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-15,
        fit_intercept=False,
    ).fit(X, Y)
    return regression.coef_.squeeze()

def opt_logistic_tuning(
    pointest,
    X,
    Y,
    Yhat,
    weights
):
    n = Y.shape[0]
    d = X.shape[1]

    mu = safe_expit(X @ pointest)

    hessian = np.zeros((d, d))
    grads_hat = np.zeros(X.shape)
    grads = np.zeros(X.shape)
    for i in range(n):
        hessian += (
                1 / n
                * mu[i]
                * (1 - mu[i])
                * np.outer(X[i], X[i])
            )
        grads_hat[i, :] = (
                X[i, :]
                * (mu[i] - Yhat[i]) * (weights[i] - 1)
            )
        grads[i, :] = X[i, :] * (mu[i] - Y[i]) * weights[i]

    grads_cent = grads - grads.mean(axis=0)
    grad_hat_cent = grads_hat - grads_hat.mean(axis=0)
    cov_grads = (1 / n) * (
        grads_cent.T @ grad_hat_cent + grad_hat_cent.T @ grads_cent
    )

    var_grads_hat = np.cov(grads_hat.T)
    
    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    h = inv_hessian[0, :]
    num = h @ cov_grads @ h
    denom = 2 * h @ var_grads_hat @ h
    lam = num / denom
    
    return lam


def logistic_cov(
    pointest,
    X,
    Y,
    Yhat,
    weights,
    lam=1
):
    n = Y.shape[0]
    d = X.shape[1]

    mu = safe_expit(X @ pointest)
    weights_mat = np.array([weights] * d)

    hessian = np.zeros((d, d))
    grads_hat = np.zeros(X.shape)
    grads = np.zeros(X.shape)
    for i in range(n):
        hessian += (
                1 / n
                * mu[i]
                * (1 - mu[i])
                * np.outer(X[i], X[i])
            )
        grads_hat[i, :] = (
                X[i, :]
                * (mu[i] - Yhat[i])
            )
        grads[i, :] = X[i, :] * (mu[i] - Y[i])

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    var = np.cov(grads.T*weights_mat + lam * grads_hat.T  - lam * grads_hat.T * weights_mat).reshape(d, d)
    return inv_hessian @ var @ inv_hessian


def active_logistic_pointestimate(
    X,
    Y,
    Yhat,
    weights,
    lam=None,
    coord=None,
    optimizer_options=None
):
    n = Y.shape[0]
    d = X.shape[1]
    
    # Set default optimizer options
    if optimizer_options is None:
        optimizer_options = {"ftol": 1e-15, "maxiter": 1000}
    if "ftol" not in optimizer_options:
        optimizer_options["ftol"] = 1e-15
    if "maxiter" not in optimizer_options:
        optimizer_options["maxiter"] = 1000

    # Initialize theta using weighted logistic regression
    theta = (
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-15,
            fit_intercept=False,
        )
        .fit(X[np.where(weights)], Y[np.where(weights)])
        .coef_.squeeze()
    )
    if len(theta.shape) == 0:
        theta = theta.reshape(1)

    # Pre-compute common terms
    X_theta = X @ theta
    expit_X_theta = safe_expit(X_theta)
    
    def rectified_logistic_loss(_theta):
        # Compute X_theta once
        X_theta = X @ _theta
        expit_X_theta = safe_expit(X_theta)
        
        # Vectorized loss computation
        loss = (
            lam * np.mean(-Yhat * X_theta + safe_log1pexp(X_theta)) -
            lam * np.mean(weights * (-Yhat * X_theta + safe_log1pexp(X_theta))) +
            np.mean(weights * (-Y * X_theta + safe_log1pexp(X_theta)))
        )
        return loss

    def rectified_logistic_grad(_theta):
        # Compute X_theta and expit once
        X_theta = X @ _theta
        expit_X_theta = safe_expit(X_theta)
        
        # Vectorized gradient computation
        grad = (
            lam * X.T @ (expit_X_theta - Yhat) / n -
            lam * X.T @ (weights * (expit_X_theta - Yhat)) / n +
            X.T @ (weights * (expit_X_theta - Y)) / n
        )
        return grad

    # Add early stopping callback
    class Callback:
        def __init__(self):
            self.iterations = []
            self.last_loss = None
            
        def __call__(self, xk):
            current_loss = rectified_logistic_loss(xk)
            if self.last_loss is not None:
                if abs(current_loss - self.last_loss) < optimizer_options["ftol"] * 10:
                    return True
            self.last_loss = current_loss
            self.iterations.append(current_loss)
            return False
    
    callback = Callback()
    callback.last_loss = rectified_logistic_loss(theta)

    # Optimize with early stopping
    pointest = minimize(
        rectified_logistic_loss,
        theta,
        jac=rectified_logistic_grad,
        method="L-BFGS-B",
        tol=optimizer_options["ftol"],
        options=optimizer_options,
        callback=callback
    ).x

    return pointest

def inv_hessian_col_imputed(X, Yhat):
    n = Yhat.shape[0]
    d = X.shape[1]
    imputed_pointest = logistic(X, (Yhat > 0.5))
    mu = safe_expit(X @ imputed_pointest)
    hessian = np.zeros((d, d))
    for i in range(n):
        hessian += (
            1 / n
            * mu[i]
            * (1 - mu[i])
            * np.outer(X[i], X[i])
        )
    inv_hessian = np.linalg.inv(hessian)
    return inv_hessian[:, 0]
