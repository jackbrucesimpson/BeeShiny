/**
 *
 */

#include "BeeTag.h"

BeeTag::BeeTag (int id, 
                cv::Point2f initial_location, 
                int initial_frame,
                int tag) :
    identity (id),
    last_location (initial_location),
    last_frame (initial_frame),
    tag_type (tag)
{
    frames.push_back (initial_frame);
    locations.push_back (initial_location);
    tags.push_back (tag);
    distance_from = initial_location;

}

void
BeeTag::add_point (cv::Point2f location,
                        int frame,
                        int classification)
{
    locations.push_back (location);
    frames.push_back (frame);
    last_location = location;
    last_frame = frame;
    tags.push_back (classification);
    tag_type = classification;
}

void
BeeTag::clear (void)
{
    locations.clear ();
    frames.clear ();
    tags.clear ();
}

void
BeeTag::delete_bee (void)
{
    deleted = true;
}

bool
BeeTag::is_deleted (void) const
{
    return deleted;
}

int
BeeTag::get_id (void) const
{
    return identity;
}

int
BeeTag::get_tag_type (void) const
{
    return tag_type;
}

std::vector<int>&
BeeTag::get_tags (void)
{
    return tags;
}

const cv::Point2f&
BeeTag::get_last_location (void) const
{
    return last_location;
}

int
BeeTag::get_last_frame (void) const
{
    return last_frame;
}

std::vector<cv::Point2f>&
BeeTag::get_locations (void)
{
    return locations;
}

std::vector<int>&
BeeTag::get_frames (void)
{
    return frames;
}

/* vim:ts=4:sw=4:sts=4:expandtab:cinoptions=(0
 */
