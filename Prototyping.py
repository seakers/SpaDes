import cProfile
import pstats
import io
import ConfigurationDesign

def profile_script():
    # Import the script to be profiled

    # Create a cProfile object
    pr = cProfile.Profile()
    pr.enable()  # Start profiling

    # Call the main function or the code you want to profile
    ConfigurationDesign.main()  # Replace with the actual function to run

    pr.disable()  # Stop profiling

    # Save the profiling results to a text file
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    
    with open('profiling_results_slow.txt', 'w') as f:
        f.write(s.getvalue())

if __name__ == "__main__":
    profile_script()