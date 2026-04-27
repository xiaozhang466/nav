/*
 * This file is part of lslidar_c16 driver.
 *
 * The driver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The driver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the driver.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <ros/ros.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <lslidar_c16_decoder/lslidar_c16_decoder.h>

// 时间戳共享内存结构 (与 FAST-LIVO 兼容)
struct time_stamp {
  int64_t high;
  int64_t low;
};
extern struct time_stamp *g_timeshare_ptr;

int main(int argc, char** argv) {
    ros::init(argc, argv, "lslidar_c16_decoder_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // 创建 timeshare 共享内存文件
    const char *user_name = getlogin();
    if (user_name != nullptr) {
        std::string path_for_time_stamp = "/home/" + std::string(user_name) + "/timeshare";
        int fd = open(path_for_time_stamp.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
        if (fd != -1) {
            lseek(fd, sizeof(time_stamp), SEEK_SET);
            write(fd, "", 1);
            g_timeshare_ptr = (time_stamp *)mmap(NULL, sizeof(time_stamp),
                                                  PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (g_timeshare_ptr != MAP_FAILED) {
                g_timeshare_ptr->high = 0;
                g_timeshare_ptr->low = 0;
                ROS_INFO("Created timeshare file: %s", path_for_time_stamp.c_str());
            } else {
                ROS_WARN("Failed to mmap timeshare file");
                g_timeshare_ptr = nullptr;
            }
            close(fd);
        } else {
            ROS_WARN("Failed to create timeshare file: %s", path_for_time_stamp.c_str());
        }
    } else {
        ROS_WARN("Failed to get username for timeshare file");
    }

    lslidar_c16_decoder::LslidarC16DecoderPtr decoder(
                new lslidar_c16_decoder::LslidarC16Decoder(nh, pnh));

    if (!decoder->initialize()) {
        ROS_INFO("Cannot initialize the decoder...");
        return -1;
    }

    ros::spin();

    // 清理共享内存
    if (g_timeshare_ptr != nullptr && g_timeshare_ptr != MAP_FAILED) {
        munmap(g_timeshare_ptr, sizeof(time_stamp));
    }

    return 0;
}
