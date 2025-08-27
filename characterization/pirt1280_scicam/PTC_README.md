# Photon Transfer Curve for the PIRT 1280SciCam
Lab data taken @ MIT in the QE closet, on night of 2025-07-29

## Data Description

### Location
Data is in the WINTER box folder, in `WINTER - Public Data Workspace/pirt_camera`

### Raw Data File Structure
The data for the PTC ramp is in `lab_testing/ramps/20250729_1050` which means it was taken on 20250729 (and into the early hours of 20250730) and the light frames are taken with a 1050 nm optical filter with 10 nm bandwidth.

The lamp/geometry was unchanged for the full ramp set, and the exposure time was changed, from 0.0001 s to 5.0 s. At each exposure time multiple images were taken and saved as a single .FITS stack. This means each stack has a variable number of layers, somewhere up to 25. Each exposure time has a directory `exptime_<EXPTIME.XXXX>` with a `darks` and `lights` subdirectory. Each image is named `scicam_stack_<UTC ISO TIME>.fits`.

## Analysis Steps
The analysis follows this procedure:
1. Load the top level ramp directory and crawl through all subdirectories to find fits files
2. Load all fits files found and then build a pandas dataframe holding:
    - filename: FITS filename (scicam_stack_20250730T034016.fits)
    - filepath: full FITS filepath (~/data/path/to/file/scicam_stack_20250730T034016.fits)
    - frame_type: type of frame, eg "light" or "dark". 
        - ideally this is pulled from the header, but for this dataset it must come from whether the image is in a "darks" or "lights" folder
    - exposure_time: exposure time in s
        - from header EXPTIME keyword
    - frame_time: frame time in s (typically exptime + 0.1 s, eg 1.1 s for a 1.0 s exptime)
        - from header FRMTIME keyword
    - filter_center: filter center wavelength in nm (eg 1050)
        - from header FILTER keyword
    - filter_bw: filter bandwidth in nm (eg 10)
        - this parses the header to get the only selected filter. the header has these entries:
            FW1FILT = 'FBH 1050-10'        / Current filter in fw1                          
            FW2FILT =  / Current filter in fw2                                              
            FW3FILT = 'EMPTY   '           / Current filter in fw3                          
            FW4FILT =  / Current filter in fw4                         
          only one of FW1/FW2/FW3/FW4 should be not "" or "EMPTY", and the "FBH 1050-10" should be parsed to get the bandwidth (10) in nm
    - temperature: FPA temperature in C
        - from header TMP_CUR
    - num_frames: number of frames in the stack
        - recorded by seeing how many layers there are in the loaded FITS image
3. Loop through all the available exposure times and make combined/flattened products at each exposure time:

    - Combine the frames within the stack for each exposure time
        - using the average, but also build in the ability to do median stacking as well
        - remove bad/outlier layers that are more than N std devs from the median layer using utils.image_stack_processing.remove_outlier_layers(stack, N)
        - do this for both the light and dark
        - Save as f`{top_dir}/processed/combined/{exptime:.4f}_{frame_type}_{combination_method}_{num_frames}_layers_combined.fits`

    - Make dark subtracted combined frames
        - eg, combined/flattened/outlier-removed light - combined/flattened/outlier-removed dark
        - make sure that the light and dark frames have the same number of layers and remove layers to make sure they do
        - might need to enforce that all dark-subtraced combined frames have the same number of layers but I'm not sure if that's true from a statistics perspective
        - Save as f`{top_dir}/processed/dark_subtracted/{exptime:.4f}_{frame_type}_{combination_method}_{num_frames}_layers_darksub.fits`

    - Light stack with each frame dark subtracted
        - eg raw light frame - average or median dark frame after outlier rejection of the dark frames
        - Save as f`{top_dir}/processed/dark_subtracted_stack/{exptime:.4f}_{frame_type}_stack_{num_frames}_layers_darksub.fits`
5. Load in all the combined dark subtracted light stacks, and make a new pandas dataframe for the processed data with:
    - exposure_time
    - filename
    - filepath
    - num_frames
    - data (eg full stack of data)
    - use remove_outlier_layers and add:
        - num_frames_cleaned
        - outlier_mask
    - cleaned_data (fulls stack with outlier layers removed)
    - mean_frame: pixel-wise mean of the cleaned_data stack (`np.nanmean(cleaned_data, axis = 0)`)
    - var_frame: pixel-wise variance frame of the cleaned_data stack (`np.nanvar(cleaned_data, axis = 0`)


6. Reject outlier stacks in an iterative approach based on the var vs mean plot:
    - df_filtered = remove_outliers_iter(df_processed, n_iters, n_std, mean_min, mean_max)
    - make all_means = processed_df["mean_frame"] and all_vars = processed_df["var_frame"]
    - plot all_means against all_vars, and calculate the slope of this plot, with means within some range (eg 20-80% of full well/saturation value for mean). Reject points that are outside N-std.
    - do this N_iters times to reject bad points
    - then mark all rows in the df that are "bad" in a new "outlier" column and return this new dataframe

7. Fit the pixel-wise photon transfer curve:

    ```python:
    # array of means, N, H, W = shape(arr_mean), where N  is the number of
    # exposure times that have data that passed the iterative cuts
    arr_mean = df_filtered["mean_frame"]
    arr_var = df_filtered["var_frame"]

    # quick way to calculate the pixel-wise slope:
    x_mean = np.nanmean(arr_mean, axis=0)
    y_mean = np.nanmean(arr_variance, axis=0)

    print(f"x_mean.shape: {x_mean.shape}")
    print(f"y_mean.shape: {y_mean.shape}")

    diff_x = arr_mean - x_mean
    diff_y = arr_variance - y_mean

    num = np.nansum(diff_x * diff_y, axis=0)
    den = np.nansum(diff_x * diff_x, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        slope_image = num / den
        slope_image[den == 0] = np.nan

    intercept_image = y_mean - slope_image * x_mean

    gain_image = slope_image ** -1

    return slope_image, intercept_image, gain_image

8. Plot the histogram of the slope image:
    ```python:
    # previous approach:
    # plot a histogram of all the calculated gains
    fig, ax = plt.subplots()
    hist = np.histogram(np.ravel(slope_image), bins=1000,range=(0, 1))
    ax.semilogy(hist[1][:-1], hist[0], '.', label="Data")
    ax.set_title("Slope histogram")

    # calculate the median gain:
    median_gain = np.nanmedian(np.ravel(gain_image))
    print(f"median gain = {median_gain:0.3f}")
    print(f"1/median gain = {1/median_gain:0.3f}")
    # plot a vertical dashed line at the median gain
    #ax.axvline(median_gain, color='r', linestyle='--', label = f"Median Gain={median_gain:0.3f}")
    ax.set_xlabel("Inv. Gain: ADU/e")
    ax.set_ylabel("Number of pixels")

    # fit a gaussian to the histogram
    xhist = hist[1][:-1]
    yhist= hist[0]
    mean_guess = xhist[np.argmax(yhist)]#np.nanmedian(gain_image)
    std_guess = 0.1#np.nanstd(gain_image)


    # fit a gaussian to the histogram
    hist_clipcond = (yhist>100)
    xhist_clip = xhist[hist_clipcond]
    yhist_clip = yhist[hist_clipcond]
    #ax.plot(xhist_clip,yhist_clip, label=f"Data to Fit")

    # make a spline fit to the clipped part of the histogram, using scipy splines
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(xhist_clip, yhist_clip)
    xhist_fine = np.linspace(xhist_clip[0], xhist_clip[-1], 1000)
    yhist_fine = cs(xhist_fine)
    ax.plot(xhist_fine, yhist_fine, label="Spline Fit")

    # get the peak of the fit
    peak = np.max(yhist_fine)
    peak_index = np.argmax(yhist_fine)
    peak_x = xhist_fine[peak_index]
    print(f"Peak Slope = {peak_x:0.3f}")
    # plot the peak gain
    ax.axvline(peak_x, color='k', linestyle='--', label = f"Peak Slope={peak_x:0.3f}->Gain={1/peak_x:0.3f} e/ADU")

    # FWHM: get it empirically from the spline fit
    fwhm = 2*np.abs(peak_x - xhist_fine[np.argmax(yhist_fine > peak/2)])
    std_from_fwhm = fwhm/2.355
    print(f"FWHM = {fwhm:0.3f}")
    print(f"std from FWHM = {std_from_fwhm:0.3f}")

    # plot a vertical line with the nominal gain
    nominal_gain = 4.2 # e/ADU
    nominal_slope = 1/nominal_gain
    print(f"Nominal Gain = {nominal_gain:0.3f}")
    print(f"Nominal Slope = {nominal_slope:0.3f}")
    ax.axvline(nominal_slope, color='g', linestyle='--', label = f"Nominal Slope={nominal_slope:0.3f}")


    # plot a vertical line at the FWHM
    ax.axvline(peak_x + fwhm/2, color='b', linestyle='--', label = f"FWHM={fwhm:0.3f}")
    ax.axvline(peak_x - fwhm/2, color='b', linestyle='--')

    # plot a vertical line at +/- N sigma from the peak
    N = 3
    ax.axvline(peak_x + N*std_from_fwhm, color='m', linestyle='--', label = f"{N} sigma={N*std_from_fwhm:0.3f}")
    ax.axvline(peak_x - N*std_from_fwhm, color='m', linestyle='--')
    # turn on the legend
    ax.legend()
    plt.show()

    ```
    
