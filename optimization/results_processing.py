''' Plot the results of the optimization. '''
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve
from intraday_utils import get_gt, get_ground_truth_pg_pb, get_gt_battery_evolution
import numpy as np
from datetime import datetime, timedelta
import os

colors = ['#43AA8B', '#ffb000', '#fe6100', '#dc267f', '#785ef0', '#648fff']
colors = plt.cm.viridis(np.linspace(0, 1, 24))


def postprocess_results(model, time_frame):
    ''' Postprocess the results of the optimization. '''

    plot_battery_evolution(model, time_frame)

    #plot_power_exchange(model, time_frame)

    plot_probabilistic_power_schedule(model, time_frame)

def plot_battery_evolution(model, time_frame):
    ''' Plots the optimal battery evolution over time. '''

    e_exp = list(model.model.e_exp.get_values().values())
    e_nominal = list(model.model.e_nom.get_values().values())
    e_prob_max = list(model.model.e_max.get_values().values())
    e_prob_min = list(model.model.e_min.get_values().values())

    e_max = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_max)]
    e_min = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_min)]

    # to compare it to the actual battery states obtained by the Intra-Day model
    # day_1
    Intra_Day_gt_day1 = [7.0, 4.50935454076485, 2.4403113183721343, 1.4151473543507112, 2.518546128831651, 4.019169476394423, 7.064469377276281, 9.982095948285696, 12.108162389219148, 13.142813554516678, 13.419933576690035, 12.05427179326, 10.974145948588864, 9.834933308661622, 8.783865419436754, 7.811232109628836, 6.927951977235979, 5.996906915129961, 5.205411996975072, 4.464532761960442, 3.673623696610867, 2.87982293321815, 2.0157735539262, 1.1679489489324235, 0.4242051264496537, 5.0444343851346884e-08, 1.7044594466812752e-06, 0.03397435296905605, 0.4180286232805428, 1.6891208978433714, 3.259357167069003, 4.872888166292924, 7.051357496123029, 7.846768888303665, 9.035151644248456, 9.288169877554576, 9.288183327700528, 9.314775810713819, 9.338225777050216, 9.151682990361492, 8.850892775828006, 8.636562992238092, 8.495414603123521, 8.344716578757012, 8.245419434641601, 8.156891981973171, 8.104734946458002, 8.045721129584077]
    Intra_Day_gt_day2 = [7.0, 3.8364435311097407, 1.251588817648649, -1.1769558549978854e-08, 0.025710302091443902, 0.178936967055367, 0.5158970607764553, 1.3992996745005981, 2.7196581669937445, 3.7121894015243364, 4.382984874459428, 4.611100216899688, 4.970082870765765, 4.76019236169267, 4.252564799721043, 3.7869433277522204, 3.3070429302793682, 2.7949499580005472, 2.228479683847687, 1.7826153287844733, 1.4427916749569611, 1.0502570660176223, 0.6956610310359019, 0.41461592727455426, 0.09667852595490012, 3.769745428663929e-08, 1.3482989491055153e-05, 0.022599156636440836, 0.2967404543458639, 1.570176331600504, 3.9394234700411803, 6.516660575297727, 8.820351885946904, 10.464651007782123, 11.563804021027307, 11.98263381206897, 12.123140054231694, 12.123153507561838, 12.136737772909148, 12.1159746121174, 12.020130289583106, 11.793370142613407, 11.602296914403466, 11.49084974043876, 11.39285098714657, 11.295692465555486, 11.225465607706358, 11.14257748040307]
    Intra_Day_gt_day3 = [7.0, 3.7538886955870847, 1.123365292780621, 3.0120774485720148e-09, 0.08003671631721543, 0.03795390158763033, 0.23188158212806265, 1.3489233403499592, 3.3975654451838957, 4.546998789618672, 6.39091869731251, 6.615148332859407, 7.459115576919953, 8.185408242534214, 7.388682659799128, 6.453981586907194, 5.386544024981871, 4.475290518791197, 3.662405342789187, 2.881410861038184, 2.283377745685987, 1.639122464505237, 0.9713241772521763, 0.36153143244188823, 2.1027671655987623e-08, 5.193386118968336e-08, 1.3497607971159913e-05, 0.010215993930065012, 0.6323822609474038, 1.2712029124892383, 2.219939462127468, 4.373861673697865, 6.725731716350974, 8.807794641828595, 10.558394702902987, 11.428996015022301, 11.51791331391392, 11.517926769118016, 11.526655424848457, 11.510761302012662, 11.213611641910276, 10.898673887167035, 10.674549435786675, 10.518491015202931, 10.364368124921146, 10.159048391749039, 10.020170796214435, 9.779683756403356]


    # get ground truth
    pl_gt = get_gt(time_frame)
    gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
    e_gt = get_gt_battery_evolution(model, gt_pb)
    

    time_e = [str(t) for t in model.model.time_e]
    ordered_time_e = model.model.time_e.ordered_data()

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_e, e_nominal, '-', color=colors[0], linewidth=1.5, label='Nominal Battery State')
    ax.plot(time_e, e_gt, '-', color='red', linewidth=1.5, label='Actual Battery State')
    #ax.plot(time_e, Intra_Day_gt_day3[:25], color='gray', linewidth=1.5, label='Actual Battey State by Intra-Day') # to plot associated intra-day policy
    #ax.plot(time_e, e_exp, linewidth=2, color='navy', label='Expected Battery State')
    #ax.plot(time_e, e_min, linewidth=1, color=colors[0])
    #ax.plot(time_e, e_max, linewidth=1, color=colors[0])

    ax.axhline(y=model.e_limit_max, color='k', linestyle='--', linewidth='2', label='Battery Limits')
    ax.axhline(y=model.e_limit_min, color='k', linestyle='--', linewidth='2')

    ax.fill_between(time_e, e_min, e_max, color=colors[0], alpha=0.2)


    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, ordered_time_e)))
    plt.xticks(np.arange(0, len(e_max), 2), rotation=45)
    plt.ylim([-0.5, model.e_limit_max + 0.6])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.94), bbox_transform=ax.transAxes)
    plt.ylabel('Battery Storage [kWh]')
    file_path = get_file_path('battery_evolution.png')
    plt.savefig(file_path, dpi=200)
    #plt.show()

def plot_power_exchange(model, time_frame):
    ''' Plots the power exchange over time.'''

    pg_nom = list(model.model.pg_nom.get_values().values())
    pb_nom = list(model.model.pb_nom.get_values().values())
    pl = [model.model.pl[t] for t in model.model.time]
    pl_gt = get_gt(time_frame)

    pg_exp_low_cond = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value / model.model.prob_low[t].value for t in model.model.time]
    pg_exp_high_cond = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value / model.model.prob_high[t].value for t in model.model.time]

    prob_low = list(model.model.prob_low.get_values().values())
    prob_high = list(model.model.prob_high.get_values().values())

    
        
    time = list(model.model.time)

    plt.figure(figsize=(10, 6))
    plt.plot(time, pb_nom, linewidth=4, label='Nominal Battery Power')
    plt.plot(time, pl, linewidth=4, label='Expected Prosumption')
    plt.plot(time, pl_gt, linewidth=4, label='Ground Truth prosumption')
    plt.plot(time, pg_nom, linewidth=4, label='Grid Power')
    #plt.plot(time, pg_exp_low_cond, linewidth=1, label='Expectation of downward deviations from Grid Power (Conditional)', color='navy')
    #plt.plot(time, pg_exp_high_cond, linewidth=1, label='Expectation of upward deviations from Grid Power (Conditional)', color='navy')
    #plt.plot(time, prob_low, linewidth=1, label='Probability of downward deviations from Grid Power', color='magenta')
    #plt.plot(time, prob_high, linewidth=1, label='Probability of upward deviations from Grid Power', color='magenta')
    plt.grid()
    plt.legend()
    plt.title('Power Exchange')
    plt.xlabel('Time')
    plt.ylabel('Power [kW]')
    file_path = get_file_path('power_exchange.png')
    plt.savefig(file_path, dpi=200)
    # plt.show()

def plot_probabilistic_power_schedule(model, time_frame, quantiles=[0.05, 0.95]):
    ''' Plot the probabilistic power schedule. '''

    time = [str(t.hour) for t in model.model.time]
    pg_nom = list(model.model.pg_nom.get_values().values())

    quant_low, quant_high  = quantiles

    # Compute the quantiles of the prosumption uncertainty
    prosumption_low = []
    prosumption_high = []
    for t in model.model.time:
        # Function with form cdf(x) - quantile = 0
        if model.probability_distribution_name == 'sum-2-gaussian-distributions':
            func_temp_low = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t],n=10) - quant_low
            func_temp_high = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t],n=10) - quant_high
        else:
            func_temp_low = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t]) - quant_low
            func_temp_high = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t]) - quant_high

        prosumption_low_temp = fsolve(func_temp_low, x0=-0.5)[0]
        prosumption_high_temp = fsolve(func_temp_high, x0=0.5)[0]

        prosumption_low.append(prosumption_low_temp)
        prosumption_high.append(prosumption_high_temp)

    # Compute the quantile values of the truncated distribution (i.e. the grid power uncertainty)
    pg_truncated_quantile_low = []
    pg_truncated_quantile_high = []
    for i in range(len(prosumption_low)):
        pg_trunc_low_temp = prosumption_low[i] - model.model.x_low[model.model.time.at(i+1)].value
        pg_trunc_high_temp = prosumption_high[i] - model.model.x_high[model.model.time.at(i+1)].value

        pg_truncated_quantile_low.append(pg_trunc_low_temp)
        pg_truncated_quantile_high.append(pg_trunc_high_temp)

    # Shift the uncertaintites to represent quantiles of the deviations from the nominal grid power
    pg_quantile_low = [pg_nomi + deviation if deviation < 0 else pg_nomi for pg_nomi, deviation in zip(pg_nom, pg_truncated_quantile_low)]
    pg_quantile_high = [pg_nomi + deviation if deviation > 0 else pg_nomi for pg_nomi, deviation in zip(pg_nom, pg_truncated_quantile_high)]

    # Unconditional Expected Deviations
    pg_exp_low = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value for t in model.model.time]
    pg_exp_high = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value for t in model.model.time]

    # Conditional ExpectedDeviations
    pg_exp_low_cond = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value / model.model.prob_low[t].value for t in model.model.time]
    pg_exp_high_cond = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value / model.model.prob_high[t].value for t in model.model.time]

    # get ground truth
    pl_gt = get_gt(time_frame)
    gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
    gt_pg = list(gt_pg)

    # Ensure that the conditional expected deviations are within the quantiles for visualization purposes.
    # With specifying the quantiles, we limit the range that interests us. Thus, the conditional expected deviations
    # should be within the quantiles. For highly asymmetrical distributions, this could not hold. Thus, for asymmetrical
    # PDFs, either the quantiles should be large (5%-95% or 1%-99%) or the following 5 lines should be commented out.
    # Otherwise, a plot that is easily misinterpreted could be generated.
    for i in range(len(pg_exp_low_cond)):
        if pg_exp_low_cond[i] < pg_quantile_low[i]:
            pg_exp_low_cond[i] = pg_quantile_low[i]
        if pg_exp_high_cond[i] > pg_quantile_high[i]:
            pg_exp_high_cond[i] = pg_quantile_high[i]

    time.append(str(int(time[-1])+0.95))  # Add a last time point for the step plot.
    pg_nom.append(pg_nom[-1])  # Repeat the last y-value (nominal grid power) for the step plot.
    pg_exp_low_cond.append(pg_exp_low_cond[-1])  
    pg_exp_high_cond.append(pg_exp_high_cond[-1])
    pg_quantile_low.append(pg_quantile_low[-1])
    pg_quantile_high.append(pg_quantile_high[-1])
    gt_pg.append(gt_pg[-1])


   

    # Plot the results
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(time, pg_nom, label='Nominal Grid Power', color=colors[0], linewidth=1.5, where='post')
    ax.step(time, gt_pg, label='Actual Grid Power', linewidth=1.5, where='post', color='red')

    
    #ax.step(time, pg_exp_low_cond, label='Expectation of Deviations', color='mediumblue', linewidth=2, where='post')
    #ax.step(time, pg_exp_high_cond, color='mediumblue', linewidth=2, where='post')


    #ax.step(time, np.ravel(pg_quantile_low), '--', label=f'{int(100*quant_low)} - {int(100*quant_high)}% Quantile', color=colors[0], linewidth=1.5, where='post')
    #ax.step(time, np.ravel(pg_quantile_high), '--', color=colors[0], linewidth=1.5, where='post')
    ax.fill_between(time,np.ravel(pg_quantile_low), np.ravel(pg_quantile_high), color=colors[0], alpha=0.2,step='post')
    # to plot probabilities in color
    cmap = plt.get_cmap('PuRd') # 'viridis'
    probs_low = list(model.model.prob_low.get_values().values())
    probs_high = list(model.model.prob_high.get_values().values())
    #norm = mcolors.Normalize(vmin=0.0, vmax=0.6) # Specify the range of the colormap => for symmetric pdfs, 0.5 is the maximum.
    #for i in range(len(time) - 1):
    #    color_value_low = cmap(norm(probs_low[i]))
    #    color_value_high = cmap(norm(probs_high[i]))
    #    ax.fill_between(time[i:i+2], pg_nom[i:i+2], pg_quantile_low[i:i+2], color=color_value_low, alpha=1.0, step='post')
    #    ax.fill_between(time[i:i+2], pg_nom[i:i+2], pg_quantile_high[i:i+2], color=color_value_high, alpha=1.0, step='post')

    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax)
    #cbar.set_label('Probability of Deviations', rotation=270, labelpad=15)

    #plt.ylim([-8.5, 4.5])
    #plt.xlim([-0.8, len(probs_low) + 0.3])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=-6.2)  # This ensures the y-axis extends to at least -6
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)

    ordered_time = model.model.time.ordered_data()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, ordered_time)))
    plt.xticks(np.arange(0, len(probs_low), 2), rotation=45)

    plt.legend(loc='upper left')
    plt.ylabel('Grid Power [kW]')
    file_path = get_file_path('dispatch_schedule.png')
    plt.savefig(file_path, dpi=200)
    plt.show()

    pass


def custom_x_axis_formatter(x, pos, time):
    ''' Custom formatter for the x-axis. '''
    # Convert position (x) to index and get the corresponding timestamp
    index = int(x) if x < len(time) else len(time) - 1
    date = time[index]  # Access the corresponding Timestamp
    if date.hour == 0 or index == 0:  # First tick of the day
        return date.strftime('%H:%M')  # Show "DD/MM HH"
        # OLD: show DD/MM/YYYY, but then it was overlapping
        #return date.strftime('%d/%m/%Y\n%H:%M')  # Show "DD/MM HH"
    else:
        return date.strftime('%H:%M')  # Otherwise show "HH:MM"


def get_file_path(filename):
    ''' Get file path to store plots in a temporary folder for logging in mlflow. '''
    temp_folder = os.path.join(os.getcwd(), 'temporary_folder')
    os.makedirs(temp_folder, exist_ok=True)

    return os.path.join(temp_folder, filename)


def validate_expected_values(model):
    ''' In the model, the expected value of the battery power uncertainty should match the one of the grid (see paper 
    Equation 8). This function checks said condition. The most probable cause for violating this condition are poor 
    integration bounds or breakpoints in very steep PDFs. Try to increase number of bkpts or adjust the bounds. '''

    pb_tilde = model.model.pb_tilde.get_values().values()

    pb_tilde = [model.model.pb_tilde[t].value for t in model.model.time]
    pg_low = [model.model.exp_pg_low[t].value for t in model.model.time]
    pg_high = [model.model.exp_pg_high[t].value for t in model.model.time]

    pg_expected = [pg_low[i] + pg_high[i] for i in range(len(pg_low))]
  
    print("Sum of expected values for validation purposes:")
    for i in range(len(pb_tilde)):
        print(f'Sum of expected values at time {list(model.model.time)[i]}: {pb_tilde[i] + pg_expected[i]}')

def show_costs(model, time_frame):

    pl_gt = get_gt(time_frame)
    gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
    gt_pg = list(gt_pg)
    pg_plus = [x if x > 0 else 0 for x in gt_pg]
    pg_minus = [x if x < 0 else 0 for x in gt_pg]
    pg_nom = list(model.model.pg_nom.get_values().values())
    

    self_suff_costs_list = [model.c11*a**2 + model.c21*b**2 for a,b in zip(pg_plus, pg_minus)]
    ss_costs = sum(self_suff_costs_list)

    prob_low = [model.model.prob_low[t].value for t in model.model.time]
    prob_high = [model.model.prob_high[t].value for t in model.model.time]
    exp_pg_low = [model.model.exp_pg_low[t].value for t in model.model.time]
    exp_pg_high = [model.model.exp_pg_high[t].value for t in model.model.time]
    grid_costs_1 = [-model.c31*a*b for a,b in zip(prob_low, exp_pg_low)]
    grid_costs_2 = [model.c32*a*b for a,b in zip(prob_high, exp_pg_high)]
                #- model.c31_varying[t] * model.prob_low[t] * model.exp_pg_low[t] 
                #+ self.c32 * model.prob_high[t] * model.exp_pg_high[t]
                #for t in model.time) 
    grid_costs_list = [a+b for a,b in zip(grid_costs_1, grid_costs_2)]
    grid_costs = sum(grid_costs_list)
    
    start = datetime.strptime('06:00', '%H:%M')
    x_label = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_label, self_suff_costs_list, label='Self-sufficiency costs')
    ax.plot(x_label, grid_costs_list, label='Grid uncertainty costs')
    ax.plot(x_label, pg_nom, label='Grid deviation costs')
    #plt.title('Self-sufficiency and grid uncertainty costs for Day Ahead Model')
    plt.legend()
    plt.xticks(x_label[::2], rotation=45)
    
    file_path = get_file_path('ss vs grid costs.png')
    plt.savefig(file_path, dpi=200)
    

    