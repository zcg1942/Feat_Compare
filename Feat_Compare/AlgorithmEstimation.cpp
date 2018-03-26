#include "AlgorithmEstimation.hpp"
#include <fstream>
#include <iterator>
#include <cstdint>

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
    if (matches.empty())
        return false;
    
    std::vector<float> distances(matches.size());
    for (size_t i=0; i<matches.size(); i++)//遍历匹配对 计算距离
        distances[i] = matches[i].distance;
    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);//计算均值和标准偏差
    
    meanDistance = static_cast<float>(mean.val[0]);//mean是scalar类型 取一个通道
    stdDev       = static_cast<float>(dev.val[0]);
    
    return false;
}

float distance(const cv::Point2f a, const cv::Point2f b)
{
    return sqrt((a - b).dot(a-b));//点乘，得到标量
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography);


bool performEstimation
(
 const FeatureAlgorithm& alg,
 const ImageTransformation& transformation,
 const cv::Mat& sourceImage,
 std::vector<FrameMatchingStatistics>& stat//stat存放各种统计参数
)
{
    Keypoints   sourceKp;
    Descriptors sourceDesc;

    cv::Mat gray;

    if (sourceImage.channels() == 3)
    {
        cv::cvtColor(sourceImage, gray, cv::COLOR_BGR2GRAY);//转为灰度图
    }
    else if (sourceImage.channels() == 4)
    {
        cv::cvtColor(sourceImage, gray, cv::COLOR_BGRA2GRAY);//A是透明度
    }
    else if(sourceImage.channels() == 1)
    {
        gray = sourceImage;
    }

    if (!alg.extractFeatures(gray, sourceKp, sourceDesc))//得到原图像的特征点和描述子
        return false;
    
    std::vector<float> x = transformation.getX();//得到变换的参数值，存储到容器中 
    stat.resize(x.size());
    
    const int count = x.size();
    
    Keypoints   resKpReal;
    Descriptors resDesc;
    Matches     matches,inliermatches;//152行会利用RANSAC计算内点和矩阵
    
    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();//milli毫
    
    //#pragma omp parallel for private(resKpReal, resDesc, matches) schedule(dynamic, 5)
    for (int i = 0; i < count; i++)//对不同的参数取值情况遍历
    {
        float       arg = x[i];
        FrameMatchingStatistics& s = stat[i];//对应的统计参数
		//在CollectedStatistics中所有的参数的初始化为0
        
        cv::Mat     transformedImage;
        transformation.transform(arg, gray, transformedImage);//不同的transformation都继承自ImageTransformation，分别在transform中用不同的方法得到dst，有的亮度直接加scalar，有的
		//gaussionBlur，有的WarpAffine，有的resize

        if (0)
        {
            std::ostringstream image_name;//基于字符串的I/O
            image_name << "image_dump_" << transformation.name << "_" << i << ".bin";
            std::ofstream dump(image_name.str().c_str(), std::ios::binary);//基于文件的I/O
            std::copy(transformedImage.datastart, transformedImage.dataend, std::ostream_iterator<uint8_t>(dump));
        }
        cv::Mat expectedHomography = transformation.getHomography(arg, gray);//返回一个3*3的单位阵 这里的expectd应该指的是配准之后的两幅图之间的
                
        int64 start = cv::getTickCount();//开始计时
        
        alg.extractFeatures(transformedImage, resKpReal, resDesc);//得到变换后图像的特征点和描述子

        // Initialize required fields
        s.isValid        = resKpReal.size() > 0;
        s.argumentValue  = arg;
        
        if (!s.isValid)
            continue;

        alg.matchFeatures(sourceDesc, resDesc, matches);//利用源图像和变换后图像的描述子得到匹配对
		// Find one best match for each query descriptor (if mask is empty).
		//不清楚这里是怎么匹配的，这只是初始匹配，之后会通过欧氏距离和阈值比较计算正确匹配数

        int64 end = cv::getTickCount();//结束计时

        std::vector<cv::Point2f> sourcePoints, sourcePointsInFrame;
        cv::KeyPoint::convert(sourceKp, sourcePoints);//将vector<KeyPoint>与vector<point2f>之间进行转换
        cv::perspectiveTransform(sourcePoints, sourcePointsInFrame, expectedHomography);//Point2f类型也可以进行透视变换。但是不知道这里为什么要用单位阵变换一下
	

        cv::Mat homography;

        //so, we have :
        //N - number of keypoints in the first image that are also visible
        //    (after transformation) on the second image

        //    N1 - number of keypoints in the first image that have been matched.

        //    n - number of the correct matches found by the matcher

        //    n / N1 - precision
        //    n / N - recall(? )

        int visibleFeatures = 0;
        int correctMatches  = 0;
        int matchesCount    = matches.size();//匹配对数 DMatcgh类型

        for (int i = 0; i < sourcePoints.size(); i++)//遍历原图像中每一个特征点
        {
            if (sourcePointsInFrame[i].x > 0 &&
                sourcePointsInFrame[i].y > 0 &&
                sourcePointsInFrame[i].x < transformedImage.cols &&
                sourcePointsInFrame[i].y < transformedImage.rows)//特征点落在变换后图像范围之内
            {
                visibleFeatures++;//可见特征点数
            }
        }

        for (int i = 0; i < matches.size(); i++)
        {
            cv::Point2f expected = sourcePointsInFrame[matches[i].trainIdx];//trainIdx是DMatch类中的int型，这句得到匹配对在原图像中对应的特征点
            cv::Point2f actual   = resKpReal[matches[i].queryIdx].pt;//得到初始匹配对在变换后图像中对应的特征点，通过.pt得到point2f类型的点
            
            if (distance(expected, actual) < 3.0)
            {
                correctMatches++;//距离小于阈值时认为是正确匹配,但对错误的也没有剔除 剔除的工作在下一步的RANSAC中
            }
        }

		bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, inliermatches, homography);//这里本来是注释掉的，直接用会报错
		//可选用least-median或者RANSAC计算出内点对和估计出的变换矩阵，返回一个bool值

        // Some simple stat:
		//在CollectedStatistics中所有的参数的初始化为0
        s.isValid        = homographyFound;
        s.totalKeypoints = resKpReal.size();//变换后图像的特征点数目
        s.consumedTimeMs = (end - start) * toMsMul;//特征点检测和初始匹配的时间
        s.precision = correctMatches / (float) matchesCount;//正确匹配比率
        s.recall = correctMatches / (float) visibleFeatures;//特征点在两幅图中重复出现中正确匹配比率

		s.correctMatchesPercent = s.precision;
		s.percentOfMatches = (float)matchesCount / (s.totalKeypoints);
		//matchingRatio再通过以上二者的乘积计算

        
        // Compute matching statistics 这部分一开始也被注释掉了
        if (homographyFound)
        {
            cv::Mat r = expectedHomography * homography.inv();//矩阵求逆
            float error = cv::norm(cv::Mat::eye(3,3, CV_64FC1) - r, cv::NORM_INF);//求无穷大范数 得到的误差以像素为单位 norm可重载 矩阵元素绝对值的平方和再开平方
			// 知乎上 使用opencv如何进行两个特征点集之间的相似度比较的？第一个答案就是用范数

            computeMatchesDistanceStatistics(inliermatches, s.meanDistance, s.stdDevDistance);//计算配对对之间的距离均值和标准差 但是这里的匹配对是RANSAC之前还是之后呢？
            s.reprojectionError = computeReprojectionError(sourceKp, resKpReal, matches, homography);//得到配对对的距离均值，标准差 最大值 最小值组成的scalar
            s.homographyError = std::min(error, 1.0f);//直接1.0是double型，后面加f变成float型的
			//只存放小于1的错误
			
            if (0 && error >= 1)
            {
                std::cout << "H expected:" << expectedHomography << std::endl;
                std::cout << "H actual:"   << homography << std::endl;
                std::cout << "H error:"    << error << std::endl;
                std::cout << "R error:"    << s.reprojectionError(0) << ";" 
                                           << s.reprojectionError(1) << ";" 
                                           << s.reprojectionError(2) << ";" 
                                           << s.reprojectionError(3) << std::endl;
                
                cv::Mat matchesImg;
				cv::drawMatches(transformedImage,
					                   resKpReal, 
									        gray, 
									    sourceKp, 
								   inliermatches,
								   matchesImg, 
                                cv::Scalar::all(-1),
                                cv::Scalar::all(-1),
                                std::vector<char>(),
                                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                //参数里面没有int型的correctmatches http://blog.sina.com.cn/sblog_a98e39a201017pgn.html
                cv::imshow("Matches", matchesImg);
                cv::waitKey(-1);
            }
        }
    }
    
    return true;
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography)
{
    assert(matches.size() > 0);

    const int pointsCount = matches.size();
    std::vector<cv::Point2f> srcPoints, dstPoints;
    std::vector<float> distances;

    for (int i = 0; i < pointsCount; i++)
    {
        srcPoints.push_back(source[matches[i].trainIdx].pt);
        dstPoints.push_back(query[matches[i].queryIdx].pt);
    }

    cv::perspectiveTransform(dstPoints, dstPoints, homography.inv());
    for (int i = 0; i < pointsCount; i++)
    {
        const cv::Point2f& src = srcPoints[i];
        const cv::Point2f& dst = dstPoints[i];

        cv::Point2f v = src - dst;
        distances.push_back(sqrtf(v.dot(v)));
    }

    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);

    cv::Scalar result;
    result(0) = mean(0);
    result(1) = dev(0);
    result(2) = *std::max_element(distances.begin(), distances.end());
    result(3) = *std::min_element(distances.begin(), distances.end());
    return result;
}
