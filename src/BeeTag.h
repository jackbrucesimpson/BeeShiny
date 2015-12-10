/**
 * The bee tracking data
 */

#ifndef __BEE_TAG_H__
#define __BEE_TAG_H__

#include <vector>

#include <opencv2/core/core.hpp>

class BeeTag
{
    public:
        BeeTag                                          (int id,
                                                         cv::Point2f initial_location,
                                                         int initial_frame,
                                                         int tag);

        void                        add_point           (cv::Point2f new_location,
                                                         int new_frame,
                                                         int new_classification);

        int                         get_id              (void) const;
        int                         get_tag_type        (void) const;
        std::vector<int>&           get_tags            (void);
        std::vector<cv::Point2f>&   get_locations       (void);
        std::vector<int>&           get_frames          (void);
        const cv::Point2f&          get_last_location   (void) const;
        int                         get_last_frame      (void) const;
        void                        clear               (void)      ;
        void                        delete_bee          (void)      ;
        bool                        is_deleted          (void) const;
        cv::Point2f distance_from;

    private:
        int identity;
        std::vector<cv::Point2f> locations;
        std::vector<int> frames;
        std::vector<int> tags;
        cv::Point2f last_location;
        
        int last_frame;
        int tag_type;
        bool deleted = false;

};

#endif /* __BEE_TAG_H__ */

/* vim:ts=4:sw=4:sts=4:expandtab:cinoptions=(0
 */
