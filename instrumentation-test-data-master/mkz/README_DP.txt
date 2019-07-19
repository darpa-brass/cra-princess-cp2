Accelerometer data was collected by a single DAU, and the DAU produced TmNS Data Messages onto the network that contained these measurements.  The acquisition box contained 3 single-axis accelerometers.  Each accelerometer was mounted in an identical orientation, thus providing redundancy.  These three accelerometer data readings were associated with DAU acquisition ports Channel 1, Channel 2, and Channel 3.  The sensor data to the DAU came in the form of a differential pair of signals.

To collect data, SwRI placed the acquisition system (DAU, sensors, power supply) inside one of its autonomous vehicles.  The course was first driven and recorded by a driver.  Once the course was set, the autonomous driving mode was utilized to drive through the course, following the same path at the same speeds for each test run.  Traversing the course took approximately 90 seconds.  Because these particular accelerometers are only single-axis sensors, multiple test runs were conducted with the sensor box being rotated to a different orientation in order to collect data from all 3 axes.

There were 3 different acquisition box orientations: BASE, FORWARD, and RIGHT:
   * BASE - the acquisition box was sitting flat on its base, forward-facing (e.g. toggle switches toward the front of the vehicle).  See BASE_orientation.jpg
   * FORWARD - the acquisition box was rotated 90 degrees forward from the BASE configuration (e.g. base plate facing rear of vehicle, top side facing front of vehicle, toggle switches toward the floor of the vehicle).  See FORWARD_orientation.jpg
   * RIGHT - the acquisition box was rotated 90 degrees to the right from the BASE configuration (e.g. base plate facing left side of vehicle, top side facing right side of vehicle).  See RIGHT_orientation.jpg
   
There were 3 different modes tested for each orientation:
    * NORMAL - all accelerometer channels wired properly, and good data being acquired by all 3 channels.  Baseline.
    * BAD - one terminal from one differential pair was disconnected from one accelerometer (channel).
    * SWITCHED - one differential pair was miswired, having its (+) and (-) terminals switched.
    
There were 2 test runs conducted for each test mode of each orientation to show some level of repeatability.

In all, there were 18 captures taken.

For each test run, a Wireshark capture was taken to collect all TmNS Data Messages from the DAU.  The accelerometer measurement data was extracted from the individual channel data streams and placed into CSV files.  These CSV files were then utilized to plot the three channels of data over time.  Each data sample was approximately 122 microseconds apart.  All 3 channels were sampled within about 10 microseconds.  For plotting, they are assumed to be at the same time tick.

Plots were generated with the python script plot_accel_data.py.  User provides a directory location that contains 3 *.csv files, and the plotter does the rest.

For each test run, a plot was generated.  The plot naming scheme is as follows:
    plot_<O><M><R>.jpg where
       <O> - orientation: (b)ase, (f)ront, or (r)ight
       <M> - mode: (n)ormal, (b)ad, (s)witched
       (R) - run number: 1 or 2
       
    example: plot_rs1.jpg - Plot for RIGHT orientation with SWITCHED test mode, test RUN number 1.
    
A video of one test run is provided in autonomous_course_DAU_accel.mp4.

CSV files and plots are contained in the "data" folder.
    data/<orientation>/<mode>_<run>/
