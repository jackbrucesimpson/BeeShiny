/**
 *
 */

#include <iostream>
#include <thread>

#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include "BeeTracker.h"

#define UNKNOWN_TAG                 0
#define O_TAG                       1
#define I_TAG                       2
#define QUEEN_TAG                   3

#define MIN_CONTOUR_AREA            150
#define MORPH_TRANSFORM_SIZE        11
#define AVG_MAX_FRAMES              10
#define BG_SNAPSHOT_EVERY_X_FRAMES  250
#define FRAMES_BEFORE_CLEAR_MEMORY  20000
#define FRAMES_BEFORE_EXTINCTION    75
#define SEARCH_SURROUNDING_AREA     200
#define SEARCH_EXPANSION_BY_FRAME   20
#define MIN_TAG_CLASSIFICATION_SIZE 26
#define MIN_CLOSENESS_BEFORE_DELETE 15

#define MAX_THREADS                 64
#define MAX_FRAMES_PER_THREAD       128


///////////////
// Utilities //
///////////////


static float euclidian_distance (const cv::Point2f &first_tag,
                                 const cv::Point2f &second_tag);

static float
euclidian_distance (const cv::Point2f &first_tag,
                    const cv::Point2f &second_tag)
{
    const float delta_x = first_tag.x - second_tag.x;
    const float delta_y = first_tag.y - second_tag.y;
    return sqrt (pow (delta_x, 2) + pow (delta_y, 2));
}

////////////////
// BeeTracker //
////////////////

BeeTracker::BeeTracker (const std::string &input_video,
                        const std::string &output_file,
                        unsigned int threads,
                        unsigned int frames) :
    input_path (input_video),
    output_path (output_file),
    n_threads (threads),
    frames_per_thread (frames)
{
    if (n_threads > 1)
    {
        if (n_threads > MAX_THREADS)
        {
            std::cerr
                << "More than "
                << MAX_THREADS
                << " have been requested ("
                << n_threads
                << ") this is probably not intended, exiting."
                << std::endl;
            exit (1);
        }
    }
    if (frames_per_thread < 1)
    {
        std::cerr
            << "Need at least one frame per thread ("
            << frames_per_thread
            << " provided)"
            << std::endl;
        exit (1);
    }
    if (frames_per_thread > MAX_FRAMES_PER_THREAD)
    {
        std::cerr
            << "More than "
            << MAX_FRAMES_PER_THREAD
            << " have been requested ("
            << frames_per_thread
            << ") this is probably not intended, exiting."
            << std::endl;
        exit (1);
    }
    
    
    //lda.load("tag_lda.yml");
    //lda_hist.load("tag_lda_hist.yml");
    //svm.load("tag_svm.yml");
    
}

const std::string&
BeeTracker::get_input_path (void) const
{
    return input_path;
}

const std::string&
BeeTracker::get_output_path (void) const
{
    return output_path;
}

unsigned int
BeeTracker::get_n_threads (void) const
{
    return n_threads;
}

int
BeeTracker::track_bees (void)
{
    // TODO check that csv path and input path are set or set them in the
    // constructor
    // Open the input video
    //cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('H', '2', '6', '4'));
    cap.open (input_path);
    if (!cap.isOpened ())
    {
        std::cerr << "Couldn't open the video" << std::endl;
        return -1;
    }
    write_csv_header ();

    segmentation_data.push_back (SegmentationData (frames_per_thread));
    for (unsigned int i = 1; i < n_threads; i++)
    {
        segmentation_data.push_back (SegmentationData (frames_per_thread));
    }

    frame_count = 0;

    boost::timer::cpu_timer timer1;
    boost::timer::cpu_timer timer2;
    boost::timer::cpu_timer timer3;

    while (true)
    {
        // Read frames
        timer1.resume ();
        bool keep_loading = load_frames ();
        timer1.stop ();

        timer2.resume ();
        segment_frames ();
        timer2.stop ();

        timer3.resume ();
        track_frames ();
        timer3.stop ();
        if (!keep_loading)
        {
            break;
        }
    }

    std::cout << "Time spent in reading     : " << timer1.format ();
    std::cout << "Time spent in segmentation: " << timer2.format ();
    std::cout << "Time spent in tracking    : " << timer3.format ();
    // Write results
    write_csv_chunk ();
    // Close the movie
    cap.release ();

    return 0;
}

bool
BeeTracker::load_frames (void)
{
    bool success = true;
    unsigned int frames_read = 0;

    for (auto &data : segmentation_data)
    {
        data.n_frames = 0;
        if (!success)
        {
            continue;
        }
        for (unsigned int i = 0; i < frames_per_thread; i++)
        {
            success = cap.read (data.frames[i]);
            if (!success)
            {
                break;
            }

            
            if (num_averaged_images == 0)
            {
                /*
                cv::Mat gray_frame_bg;
                cv::cvtColor (data.frames[i], gray_frame_bg, CV_BGR2GRAY);
                sum_matrix = cv::Mat::zeros(gray_frame_bg.size(), CV_64FC1);
                cv::accumulateWeighted(gray_frame_bg, sum_matrix, 0.0001);
                */

                cv::Mat gray_frame_bg;
                cv::cvtColor (data.frames[i], gray_frame_bg, CV_BGR2GRAY);
                //cv::GaussianBlur (gray_frame_bg, gray_frame_bg, cv::Size (5, 5), 0);
                //pMOG(gray_frame_bg, fgMaskMOG);
                //pMOG2(gray_frame_bg, fgMaskMOG2);
                sum_matrix = cv::Mat::zeros(gray_frame_bg.size(), CV_64FC1);
                cv::accumulateWeighted(gray_frame_bg, sum_matrix, 0.1);
                //sum_matrix = gray_frame_bg.clone();

                //cv::Mat gray_frame_bg;
                //cv::cvtColor (data.frames[i], gray_frame_bg, CV_BGR2GRAY);
                //sum_matrix = gray_frame_bg.clone();
                //sum_matrix.convertTo(sum_matrix, CV_64F);
                num_averaged_images++;
                std::cout << num_averaged_images << "\n";
            }
            else if (num_frames_passed % BG_SNAPSHOT_EVERY_X_FRAMES == 0)
            {

                cv::Mat gray_frame_bg;
                cv::cvtColor (data.frames[i], gray_frame_bg, CV_BGR2GRAY);
                cv::accumulateWeighted(gray_frame_bg, sum_matrix, 0.1);
                //cv::GaussianBlur (gray_frame_bg, gray_frame_bg, cv::Size (5, 5), 0);
                //pMOG(gray_frame_bg, fgMaskMOG);
                //pMOG2(gray_frame_bg, fgMaskMOG2);

                //cv::Mat gray_frame_bg;
                //cv::cvtColor (data.frames[i], gray_frame_bg, CV_BGR2GRAY);
                //cv::accumulateWeighted(gray_frame_bg, sum_matrix, 0.01);

                //cv::Mat gray_frame_bg;
                //cv::cvtColor (data.frames[i], gray_frame_bg, CV_BGR2GRAY);
                //gray_frame_bg.convertTo(gray_frame_bg, CV_64F);
                //sum_matrix += gray_frame_bg;
                num_averaged_images++;
                std::cout << num_averaged_images << " adding\n";
            }
            
            if (num_averaged_images == AVG_MAX_FRAMES)
            {   
                /*
                cv::Mat bg;
                std::string image_file_name = "mog.png";
                pMOG.getBackgroundImage(bg);
                equalizeHist(bg, bg);
                cv::imwrite(image_file_name, bg);
                image_file_name = "mog2.png";
                pMOG2.getBackgroundImage(bg);
                equalizeHist(bg, bg);
                cv::imwrite(image_file_name, bg);
                */
                sum_matrix.convertTo(sum_matrix, CV_8UC1);
                equalizeHist(sum_matrix, sum_matrix);
                std::string image_file_name = "running.png";
                cv::imwrite(image_file_name, sum_matrix);

                //std::cout << float(num_averaged_images) << "\n";
                //sum_matrix /= float(num_averaged_images);
                /*
                sum_matrix.convertTo(sum_matrix, CV_8UC1);
                std::string image_file_name = output_path + "_";
                image_file_name = image_file_name + std::to_string(img_num_written);
                image_file_name = image_file_name + ".png";
                cv::imwrite(image_file_name, sum_matrix);
                */
                num_averaged_images = 0;
                img_num_written++;
                std::cout << num_averaged_images << " done\n";
                
            }

            num_frames_passed++;


            //std::cout << data.n_frames << " " << frames_read << " " << " " << num_averaged_images << " " << data.frames[i].cols << " " << data.frames[i].rows << "\n";
            data.n_frames++;
            frames_read++;
            
        }
    }
    return success;
}

void
BeeTracker::segment_frames (void)
{
    if (n_threads == 0)
    {
        find_countours (0);
    }
    else
    {
        // Launch the threads to find the tags in each frame
        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < n_threads; i++)
        {
            threads.push_back (std::thread (&BeeTracker::find_countours,
                                            this, i));
        }
        // Wait for threads to complete
        for (auto &thread : threads)
        {
            thread.join ();
        }
    }
}

void
BeeTracker::track_frames (void)
{
        // Track bees accross frames
        for (auto &data : segmentation_data)
        {
            if (data.n_frames == 0)
            {
                break;
            }
            for (size_t i = 0; i < data.n_frames; i++)
            {
                std::vector<cv::Point2f> &tag_locations = data.locations[i];
                std::vector<int> &tag_classifications = data.classifications[i];

                for (size_t j = 0; j < tag_locations.size (); j++)
                {
                    bool new_bee_found = identify_past_location (tag_locations,
                                                                 tag_classifications,
                                                                 j, frame_count);
                    if (new_bee_found)
                    {
                        BeeTag new_bee (bee_ids,
                                        tag_locations[j],
                                        frame_count,
                                        tag_classifications[j]);
                        all_bees.push_back (new_bee);
                        
                        bee_ids++;
                    }
                }

                // Done for this frame, clear the vectors
                tag_locations.clear ();
                tag_classifications.clear ();
                data.tags[i].clear ();

                //// This could be useful information but it should go to a log file and
                //// not be printed on every  iterations
                //// only if verbose mode or debug mode
                //std::cerr << frame_count << " " << all_bees.size () << std::endl;
                frame_count++;

                // Write results
                if (frame_count % FRAMES_BEFORE_CLEAR_MEMORY == 0)
                {
                    write_csv_chunk ();
                }
            }
        }
}

void
BeeTracker::find_countours (unsigned int thread_id)
{
    cv::Mat gray_frame;
    cv::Mat smooth_frame;
    cv::Mat thresh_frame;
    //cv::Mat thresh_shape;
    //cv::Mat new_avg_frame;
    cv::Mat close_element = cv::getStructuringElement (cv::MORPH_ELLIPSE, //ELLIPSE
                                                       cv::Size (5,
                                                                 5)); //ELLIPSE

    cv::Mat close_element2 = cv::getStructuringElement (cv::MORPH_RECT, //ELLIPSE
                                                       cv::Size (5,
                                                                 5)); //ELLIPSE

    SegmentationData &data = segmentation_data[thread_id];
    // Tag segmentation: colour to gray conversion, smoothing,closing and blocking reflection and thresholding
    for (size_t i = 0; i < data.n_frames; i++)
    {
        cv::cvtColor (data.frames[i], gray_frame, CV_BGR2GRAY);
        cv::GaussianBlur (gray_frame, smooth_frame, cv::Size (9, 9), 0);

        //cv::blur (gray_frame, smooth_frame, cv::Size (5, 5)); // 11, 11

        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        cv::Mat total_gradient;

        cv::Scharr(smooth_frame, grad_x, CV_16S, 1, 0, 1.5, 1, cv::BORDER_DEFAULT);
        cv::Scharr(smooth_frame, grad_y, CV_16S, 0, 1, 1.5, 1, cv::BORDER_DEFAULT);
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, total_gradient);
        
        cv::blur (total_gradient, total_gradient, cv::Size (11, 11));

        cv::threshold (total_gradient, thresh_frame, 140, 255, CV_THRESH_BINARY);
        //cv::threshold (gray_frame, thresh_shape, 150, 255, CV_THRESH_BINARY_INV);

        cv::morphologyEx (thresh_frame, thresh_frame, cv::MORPH_OPEN, close_element);
        

        //cv::dilate(thresh_frame, thresh_frame, close_element2);
        //dilate(opening,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)), 1)

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours (thresh_frame, contours, hierarchy,
                          CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE,
                          cv::Point (0, 0)); // CV_RETR_LIST

        int num_classified = 0;
        std::vector<int> tags_for_caffe_classification;

        for (auto &contour : contours)
        {
            if (cv::contourArea (contour) > MIN_CONTOUR_AREA)
            {
                bool needs_classification;
                cv::Mat tag_matrix;
                cv::Point2f located = locate_bee (contour, gray_frame, needs_classification, tag_matrix);
                data.locations[i].push_back (located);
                data.classifications[i].push_back(0);
                if (needs_classification)
                {
                    tags_for_caffe_classification.push_back(num_classified);
                    //std::cout << num_classified << std::endl;
                    data.tags[i].push_back(tag_matrix);
                }
              
              num_classified++;
              
            }
        
        }
        
      //std::cout << "first" << std::endl;
      //std::cout << "num matrices: " << data.tags[i].size() << " num for classification: " << tags_for_caffe_classification.size() << " num tags total " << data.classifications[i].size() << " num iterated over: " << num_classified << "\n";
      
      /*
        
        classification_lock.lock();
        
        std::vector<Prediction> predictions;
        if (data.tags[i].size() > 0)
        {
            predictions = classifier.Classify(data.tags[i]);
            
        }
        classification_lock.unlock();
      
      //std::cout << "num predictions: " << predictions.size() << "\n";

        //std::cout << "pc \n";
        for(int ii = 0; ii < predictions.size(); ii++)
        {
          
          if (predictions[ii].second > 0.9)
            {
              //std::cout << tags_for_caffe_classification[ii] << std::endl;
              data.classifications[i][tags_for_caffe_classification[ii]] = std::stoi(predictions[ii].first);
              
           }

        }
        tags_for_caffe_classification.clear();
        */
        //std::cout << "dpc \n";
      //std::cout << "done \n\n";
    }
}

void
BeeTracker::write_csv_header (void)
{
    time_t _tm = time (NULL );
    struct tm * curtime = localtime ( &_tm );

    output_stream.open (output_path);
    // Metadata as comment
    output_stream
        << "# Version: "
        << VERSION
        << " File: "
        << input_path
        << " Video date: "
        << "XXX" //// NEED to extract this from the video
        << " Processing date: "
        << asctime (curtime)
    // Header
        << "BeeID,"
        << "Tag,"
        << "Frame,"
        << "X,"
        << "Y"
        << std::endl;
}

void
BeeTracker::write_csv_chunk (void)
{
    for (size_t i = 0; i < all_bees.size (); i++)
    {

        

        std::vector<cv::Point2f> every_location_of_bee = all_bees[i].get_locations ();
        std::vector<int> all_frames_bee_present = all_bees[i].get_frames ();
        std::vector<int> all_tags_classified = all_bees[i].get_tags ();


        for (size_t j = 0; j < all_frames_bee_present.size (); j++)
        {
            output_stream
                << all_bees[i].get_id ()
                << ","
                << all_tags_classified[j]
                << ","
                << all_frames_bee_present[j]
                << ","
                << every_location_of_bee[j].x
                << ","
                << every_location_of_bee[j].y
                << std::endl;
        }

    }

    // TODO Move this code to another function?
    int count = frame_count;
    all_bees.erase (std::remove_if (all_bees.begin (), all_bees.end (), [count](BeeTag &bee) -> bool
        {
            int frames_since_last_seen = count - bee.get_last_frame ();

            if (frames_since_last_seen > FRAMES_BEFORE_EXTINCTION)
            {
                return true;
            }
            else
            {
                bee.clear ();
                return false;
            }
        }
    ), all_bees.end ());
}

cv::Point2f
BeeTracker::locate_bee (const std::vector<cv::Point> &each_contour,
                     const cv::Mat &gray_frame,
                       bool & needs_classification,
                     cv::Mat &tag_matrix)

{
    cv::RotatedRect surrounding_rectangle = cv::minAreaRect (cv::Mat(each_contour));
    cv::Point2f locate = surrounding_rectangle.center;

    //std::cout << surrounding_rectangle.size.width << " " << surrounding_rectangle.size.height << std::endl;

    if (surrounding_rectangle.size.width < MIN_TAG_CLASSIFICATION_SIZE ||
        surrounding_rectangle.size.height < MIN_TAG_CLASSIFICATION_SIZE ||
        std::abs(surrounding_rectangle.size.width - surrounding_rectangle.size.height) > 2 ||
        locate.x - 12 < 0 ||
        locate.x + 12 > gray_frame.cols ||
        locate.y - 12 < 0 ||
        locate.y + 12 > gray_frame.rows)
    {
        needs_classification = false;
        return locate;
    }

    else
    {
        cv::Rect roi = cv::Rect (locate.x - 12, locate.y - 12, 24, 24);
        cv::Mat tag_area = gray_frame (roi);
        
        cv::Mat tag_area_filtered;
        cv::medianBlur(tag_area, tag_area_filtered, 3);
        tag_area_filtered.convertTo(tag_area_filtered, CV_32FC1); // use CV_32FC3 for color images
        int kernel_size = 3;
        double sig = 1.0, th = 0.2, lm = 3.0, gm = 1.5, ps = 0;
        cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm, ps);
        cv::filter2D(tag_area_filtered, tag_area_filtered, CV_32F, kernel);
        tag_area_filtered=tag_area_filtered*0.00390625;
                
        tag_matrix = tag_area_filtered;
        needs_classification = true;
        return locate;
        
        //std::cout << tag_area_filtered.rows << ' ' << tag_area_filtered.cols << std::endl;

    }

}

bool
BeeTracker::identify_past_location (std::vector<cv::Point2f> &tag_locations,
                                    std::vector<int> &tag_classifications,
                                    size_t iterator,
                                    size_t frame_number)
{
    if (all_bees.empty ())
    {
        return true;
    }

    cv::Point2f current_tag_contour = tag_locations[iterator];
    int tag_best_match_position;
    int best_match_frames_since_last_seen = 1000;
    float closest_match_distance = 100000;
    bool found_bee_previously = false;
    bool have_to_delete_bee = false;

    for (size_t i = 0; i < all_bees.size (); i++)
    {
        cv::Point2f last_location_of_bee = all_bees[i].get_last_location ();
        int frames_since_last_seen = frame_number - all_bees[i].get_last_frame ();
        bool better_match_available = false;
        bool bee_too_close_to_other = false;
        float closeness_of_tag_to_current_contour = euclidian_distance (current_tag_contour,
                                                                        last_location_of_bee);

        if (closeness_of_tag_to_current_contour < SEARCH_SURROUNDING_AREA &&
            frames_since_last_seen < FRAMES_BEFORE_EXTINCTION &&
            !all_bees[i].is_deleted ())
        {
            for (size_t j = 0; j < tag_locations.size (); j++)
            {
                if (iterator != j)
                {
                    float closeness_to_other_tag = euclidian_distance (tag_locations[j],
                                                                       last_location_of_bee);
                    if (closeness_to_other_tag < closeness_of_tag_to_current_contour)
                    {
                        better_match_available = true;
                        break;
                    }
                    else if (closeness_to_other_tag < MIN_CLOSENESS_BEFORE_DELETE)
                    {
                        bee_too_close_to_other = true;
                    }
                }
                
            }
            if (!better_match_available &&
                closeness_of_tag_to_current_contour < closest_match_distance)
            {
                tag_best_match_position = i;
                closest_match_distance = closeness_of_tag_to_current_contour;
                best_match_frames_since_last_seen = frames_since_last_seen;
                found_bee_previously = true;
                if (bee_too_close_to_other)
                {
                    have_to_delete_bee = true;
                    bee_too_close_to_other = false;
                }
                else
                {
                    have_to_delete_bee = false;
                }
            }
        }
    }

    int expand_search_radius = (best_match_frames_since_last_seen * SEARCH_EXPANSION_BY_FRAME) + SEARCH_EXPANSION_BY_FRAME;
    if (found_bee_previously && expand_search_radius > closest_match_distance)
    {
        all_bees[tag_best_match_position].add_point (tag_locations[iterator],
                                                     frame_number,
                                                     tag_classifications[iterator]);

        if (euclidian_distance(all_bees[tag_best_match_position].distance_from, tag_locations[iterator]) > 10)
        {
            all_bees[tag_best_match_position].distance_from = tag_locations[iterator];
        }

        if (have_to_delete_bee)
        {
            all_bees[tag_best_match_position].delete_bee ();
        }
        return false;
    }
    return true;
}

/* vim:ts=4:sw=4:sts=4:expandtab:cinoptions=(0:
 */
