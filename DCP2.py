import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.colors as colors

def read_from_excel(filename, sheet_name=0, cell_range=None):
    """
    Read voltage data from Excel file
    filename: path to your Excel file
    sheet_name: name or index of the sheet (default is first sheet)
    cell_range: specific range to read (e.g., 'AA11:AL22')
    """
    if cell_range:
        # Read specific range
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols='AA:AL', 
                          skiprows=10, nrows=12, header=None)
    else:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    return df.values

def create_bipolar_contour(data, style='both', interpolation_factor=5):
    """
    Create bipolar field contour plots
    
    Parameters:
    -----------
    data : numpy array
        2D array of voltage measurements
    style : str
        'filled' for style A (filled contours only)
        'lines' for style B (filled with contour lines)
        'both' for both styles side by side
    interpolation_factor : int
        Factor for smoothing (higher = smoother gradients)
    """
    
    # Create coordinate grids
    rows, cols = data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Interpolate for smoother contours
    xi = np.linspace(0, cols-1, cols*interpolation_factor)
    yi = np.linspace(0, rows-1, rows*interpolation_factor)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Flatten the arrays for griddata
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = data.ravel()
    
    # Interpolate the data
    Zi = griddata(points, values, (Xi, Yi), method='cubic')
    
    # Find the range for symmetric color scale
    vmax = np.nanmax(np.abs(Zi))
    vmin = -vmax
    
    # Set up the plot
    if style == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        axes = [ax1, ax2]
        styles = ['filled', 'lines']
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
        styles = [style]
    
    for ax, s in zip(axes, styles):
        # Create filled contour plot
        contourf = ax.contourf(Xi, Yi, Zi, levels=20, 
                               cmap='RdBu_r', vmin=vmin, vmax=vmax,
                               extend='both')
        
        # Add contour lines for style B
        if s == 'lines':
            contour_lines = ax.contour(Xi, Yi, Zi, levels=15, 
                                      colors='black', linewidths=0.5, 
                                      alpha=0.5)
            # Add zero contour as a thicker line
            zero_contour = ax.contour(Xi, Yi, Zi, levels=[0], 
                                     colors='black', linewidths=1.5)
        
        # Add markers for electrodes (+ and -)
        # Find approximate locations of max positive and negative
        pos_idx = np.unravel_index(np.nanargmax(data), data.shape)
        neg_idx = np.unravel_index(np.nanargmin(data), data.shape)
        
        # Plot electrode markers
        ax.plot(neg_idx[1], neg_idx[0], 'ko', markersize=12)
        ax.plot(neg_idx[1], neg_idx[0], 'w-', markersize=8, markeredgewidth=2)
        
        ax.plot(pos_idx[1], pos_idx[0], 'ko', markersize=12)
        ax.plot(pos_idx[1], pos_idx[0], 'w+', markersize=8, markeredgewidth=2)
        
        # Set labels and title
        ax.set_xlabel('Column (A-L)', fontsize=12)
        ax.set_ylabel('Row (1-12)', fontsize=12)
        
        # Set x-axis labels to A-L
        ax.set_xticks(range(cols))
        ax.set_xticklabels([chr(65+i) for i in range(cols)])
        
        # Set y-axis labels to 1-12
        ax.set_yticks(range(rows))
        ax.set_yticklabels(range(1, rows+1))
        
        # Invert y-axis to match spreadsheet orientation
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set title
        if s == 'filled':
            ax.set_title('A: Filled Contours', fontsize=14, fontweight='bold')
        else:
            ax.set_title('B: Contours with Field Lines', fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(contourf, cax=cbar_ax)
    cbar.set_label('Voltage (mV)', rotation=270, labelpad=20, fontsize=12)
    
    # Add main title
    fig.suptitle('Electric Organ Discharge Field Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig

# MAIN EXECUTION
if __name__ == "__main__":

    data = read_from_excel('DCP2.xlsx', sheet_name=0, cell_range='AA11:AL22')
    
    # Create the plots
    # ================
    # Options for style: 'filled', 'lines', or 'both'
    fig = create_bipolar_contour(data, style='both', interpolation_factor=5)
    
    # Save the figure (optional)
    # fig.savefig('electric_field_analysis.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Print data statistics
    print("\nData Statistics:")
    print("Shape: {}".format(data.shape))
    print("Max voltage: {:.2f} mV".format(np.max(data)))
    print("Min voltage: {:.2f} mV".format(np.min(data)))
    print("Mean voltage: {:.2f} mV".format(np.mean(data)))
    print("Zero crossing points: {}".format(np.sum(np.diff(np.sign(data.flatten())) != 0)))