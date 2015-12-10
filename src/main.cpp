/**
 * Main
 */

#include <iostream>
#include <string>

#include "boost/program_options.hpp"

#include "BeeTracker.h"

#include <opencv2/opencv.hpp>

int
main (int argc, char *argv[])
{
    // Define and parse program options
    namespace po = boost::program_options;
    po::options_description desc ("Command line arguments");
    desc.add_options ()
        ("help,h", "Print help messages")
        ("verbose,v", "Verbose")
        ("threads,t", po::value<unsigned int> (), "Number of threads (0 for unthreaded)")
        ("frames,f", po::value<unsigned int> (), "Number of frames allocated to each thread")
        ("input,i", po::value<std::string>()->required (), "Input video file")
        ("output,o", po::value<std::string>()->required (), "Output csv file");

    po::positional_options_description positionalOptions;
    positionalOptions.add ("input", 1);
    positionalOptions.add ("output", 1);

    std::string input;
    std::string output;
    unsigned int n_threads = 0;
    unsigned int frames_per_thread = 1;

    po::variables_map vm;
    try
    {
        po::store (po::parse_command_line (argc, argv, desc), vm);

        /** --help option */
        if (vm.count ("help"))
        {
            std::cout << "Basic Command Line Parameter App"
                      << std::endl
                      << desc
                      << std::endl;
            return 0;
        }
        // throws on error, so do after help in case there are any problems
        po::notify (vm);

        if (vm.count ("input"))
        {
            input = vm["input"].as<std::string> ();
        }
        if (vm.count ("output"))
        {
            output = vm["output"].as<std::string> ();
        }
        if (vm.count ("threads"))
        {
            n_threads = vm["threads"].as<unsigned int> ();
        }
        if (vm.count ("frames"))
        {
            frames_per_thread = vm["frames"].as<unsigned int> ();
        }
    }
    catch (po::error& e)
    {
        std::cerr << "ERROR: "
                  << e.what ()
                  << std::endl
                  << std::endl
                  << desc
                  << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);
    BeeTracker tracker (input, output, n_threads, frames_per_thread);
    tracker.track_bees ();

    return 0;
}

/* vim:ts=4:sw=4:sts=4:expandtab:
 */
