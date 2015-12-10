/**k
 * The bee tracker
 */

#ifndef __BEE_TRACKER_H__
#define __BEE_TRACKER_H__

#include <fstream>
#include <string>
#include <mutex>

#include <opencv2/opencv.hpp>

#define CPU_ONLY
#include "BeeTag.h"
#include "Classifier.h"


#define VERSION             1.1


class BeeTracker
{
    public:
        BeeTracker                                  (const std::string &input_video,
                                                     const std::string &output_file,
                                                     unsigned int threads,
                                                     unsigned int frames_per_thread);

        const std::string  &get_input_path          (void) const;

        const std::string  &get_output_path         (void) const;

        int                 track_bees              (void);

        unsigned int        get_n_threads           (void) const;

    private:

        struct SegmentationData
        {
            SegmentationData (unsigned int frames_per_thread) :
                frames (frames_per_thread),
                locations (frames_per_thread),
                classifications (frames_per_thread),
                tags (frames_per_thread) {}

            std::vector<cv::Mat>                  frames;
            std::vector<std::vector<cv::Point2f>> locations;
            std::vector<std::vector<int>>         classifications;
            std::vector<std::vector<cv::Mat>>     tags;
            unsigned int n_frames = 0;
        };

        void                write_csv_header        (void);

        void                write_csv_chunk         (void);

        bool                load_frames             (void);

        void                segment_frames          (void);

        void                track_frames            (void);

        void                find_countours          (unsigned int thread_id);

        cv::Point2f         locate_bee     (const std::vector<cv::Point> &each_contour,
                                                     const cv::Mat &thresh_shape_frame,
                                                        bool & needs_classification,
                                                     cv::Mat &tag_matrix);

        bool                identify_past_location  (std::vector<cv::Point2f> &tag_locations,
                                                     std::vector<int> &tag_classifications,
                                                     size_t iterator,
                                                     size_t frame_number);

        //std::vector<int> classify_tags (std::vector<cv::Mat> current_frame_tags);


        // Options
        std::string         input_path = "";
        std::string         output_path = "";
        unsigned int        n_threads = 1;
        unsigned int        frames_per_thread = 10;
        // Frame loading stuff
        cv::VideoCapture    cap;
        std::ofstream       output_stream;
        int                 frame_count = 0;
        //std::vector<cv::Mat>    background_frames;
        int                 num_averaged_images = 0;
        int                 num_frames_passed = 0;
        int                 img_num_written = 0;
        cv::Mat             sum_matrix, averaged_image;

        /*
        cv::Ptr<cv::BackgroundSubtractor> pMOG; //MOG Background subtractor
        cv::Ptr<cv::BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
        cv::Mat fgMaskMOG; //fg mask generated by MOG method
        cv::Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
        cv::BackgroundSubtractor pMOG = createBackgroundSubtractorMOG(); //MOG approach
        cv::BackgroundSubtractor pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
        */
        //cv::Mat fgMaskMOG;
        //cv::Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
        //cv::Ptr<cv::BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
        //cv::BackgroundSubtractorMOG pMOG;//(history=, nmixtures, double backgroundRatio, double noiseSigma=0);
        //cv::BackgroundSubtractorMOG2 pMOG2;
        //pMOG2 = BackgroundSubtractorMOG2();

        //cv::LDA lda(2);

        std::string model_file = "../data/deploy_arch.prototxt";
        std::string trained_file = "../data/model.caffemodel";
        std::string label_file = "../data/labels.txt";
        Classifier classifier{model_file, trained_file, label_file};


        //int                 n_frames = 0;
        //cv::Mat             sum_matrix, averaged_image;
        // Currently tracked bees
        std::vector<BeeTag> all_bees;
        int                 bee_ids = 0;

        // Segmentation
        std::vector<SegmentationData> segmentation_data;

        std::mutex classification_lock;
};

#endif /* __BEE_TRACKER_H__ */

/* vim:ts=4:sw=4:sts=4:expandtab:cinoptions=(0:
 */
