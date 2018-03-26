#include "AlgorithmEstimation.hpp"
#include <fstream>
#include <iterator>
#include <cstdint>

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
    if (matches.empty())
        return false;
    
    std::vector<float> distances(matches.size());
    for (size_t i=0; i<matches.size(); i++)//����ƥ��� �������
        distances[i] = matches[i].distance;
    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);//�����ֵ�ͱ�׼ƫ��
    
    meanDistance = static_cast<float>(mean.val[0]);//mean��scalar���� ȡһ��ͨ��
    stdDev       = static_cast<float>(dev.val[0]);
    
    return false;
}

float distance(const cv::Point2f a, const cv::Point2f b)
{
    return sqrt((a - b).dot(a-b));//��ˣ��õ�����
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography);


bool performEstimation
(
 const FeatureAlgorithm& alg,
 const ImageTransformation& transformation,
 const cv::Mat& sourceImage,
 std::vector<FrameMatchingStatistics>& stat//stat��Ÿ���ͳ�Ʋ���
)
{
    Keypoints   sourceKp;
    Descriptors sourceDesc;

    cv::Mat gray;

    if (sourceImage.channels() == 3)
    {
        cv::cvtColor(sourceImage, gray, cv::COLOR_BGR2GRAY);//תΪ�Ҷ�ͼ
    }
    else if (sourceImage.channels() == 4)
    {
        cv::cvtColor(sourceImage, gray, cv::COLOR_BGRA2GRAY);//A��͸����
    }
    else if(sourceImage.channels() == 1)
    {
        gray = sourceImage;
    }

    if (!alg.extractFeatures(gray, sourceKp, sourceDesc))//�õ�ԭͼ����������������
        return false;
    
    std::vector<float> x = transformation.getX();//�õ��任�Ĳ���ֵ���洢�������� 
    stat.resize(x.size());
    
    const int count = x.size();
    
    Keypoints   resKpReal;
    Descriptors resDesc;
    Matches     matches,inliermatches;//152�л�����RANSAC�����ڵ�;���
    
    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();//milli��
    
    //#pragma omp parallel for private(resKpReal, resDesc, matches) schedule(dynamic, 5)
    for (int i = 0; i < count; i++)//�Բ�ͬ�Ĳ���ȡֵ�������
    {
        float       arg = x[i];
        FrameMatchingStatistics& s = stat[i];//��Ӧ��ͳ�Ʋ���
		//��CollectedStatistics�����еĲ����ĳ�ʼ��Ϊ0
        
        cv::Mat     transformedImage;
        transformation.transform(arg, gray, transformedImage);//��ͬ��transformation���̳���ImageTransformation���ֱ���transform���ò�ͬ�ķ����õ�dst���е�����ֱ�Ӽ�scalar���е�
		//gaussionBlur���е�WarpAffine���е�resize

        if (0)
        {
            std::ostringstream image_name;//�����ַ�����I/O
            image_name << "image_dump_" << transformation.name << "_" << i << ".bin";
            std::ofstream dump(image_name.str().c_str(), std::ios::binary);//�����ļ���I/O
            std::copy(transformedImage.datastart, transformedImage.dataend, std::ostream_iterator<uint8_t>(dump));
        }
        cv::Mat expectedHomography = transformation.getHomography(arg, gray);//����һ��3*3�ĵ�λ�� �����expectdӦ��ָ������׼֮�������ͼ֮���
                
        int64 start = cv::getTickCount();//��ʼ��ʱ
        
        alg.extractFeatures(transformedImage, resKpReal, resDesc);//�õ��任��ͼ����������������

        // Initialize required fields
        s.isValid        = resKpReal.size() > 0;
        s.argumentValue  = arg;
        
        if (!s.isValid)
            continue;

        alg.matchFeatures(sourceDesc, resDesc, matches);//����Դͼ��ͱ任��ͼ��������ӵõ�ƥ���
		// Find one best match for each query descriptor (if mask is empty).
		//�������������ôƥ��ģ���ֻ�ǳ�ʼƥ�䣬֮���ͨ��ŷ�Ͼ������ֵ�Ƚϼ�����ȷƥ����

        int64 end = cv::getTickCount();//������ʱ

        std::vector<cv::Point2f> sourcePoints, sourcePointsInFrame;
        cv::KeyPoint::convert(sourceKp, sourcePoints);//��vector<KeyPoint>��vector<point2f>֮�����ת��
        cv::perspectiveTransform(sourcePoints, sourcePointsInFrame, expectedHomography);//Point2f����Ҳ���Խ���͸�ӱ任�����ǲ�֪������ΪʲôҪ�õ�λ��任һ��
	

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
        int matchesCount    = matches.size();//ƥ����� DMatcgh����

        for (int i = 0; i < sourcePoints.size(); i++)//����ԭͼ����ÿһ��������
        {
            if (sourcePointsInFrame[i].x > 0 &&
                sourcePointsInFrame[i].y > 0 &&
                sourcePointsInFrame[i].x < transformedImage.cols &&
                sourcePointsInFrame[i].y < transformedImage.rows)//���������ڱ任��ͼ��Χ֮��
            {
                visibleFeatures++;//�ɼ���������
            }
        }

        for (int i = 0; i < matches.size(); i++)
        {
            cv::Point2f expected = sourcePointsInFrame[matches[i].trainIdx];//trainIdx��DMatch���е�int�ͣ����õ�ƥ�����ԭͼ���ж�Ӧ��������
            cv::Point2f actual   = resKpReal[matches[i].queryIdx].pt;//�õ���ʼƥ����ڱ任��ͼ���ж�Ӧ�������㣬ͨ��.pt�õ�point2f���͵ĵ�
            
            if (distance(expected, actual) < 3.0)
            {
                correctMatches++;//����С����ֵʱ��Ϊ����ȷƥ��,���Դ����Ҳû���޳� �޳��Ĺ�������һ����RANSAC��
            }
        }

		bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, inliermatches, homography);//���ﱾ����ע�͵��ģ�ֱ���ûᱨ��
		//��ѡ��least-median����RANSAC������ڵ�Ժ͹��Ƴ��ı任���󣬷���һ��boolֵ

        // Some simple stat:
		//��CollectedStatistics�����еĲ����ĳ�ʼ��Ϊ0
        s.isValid        = homographyFound;
        s.totalKeypoints = resKpReal.size();//�任��ͼ�����������Ŀ
        s.consumedTimeMs = (end - start) * toMsMul;//��������ͳ�ʼƥ���ʱ��
        s.precision = correctMatches / (float) matchesCount;//��ȷƥ�����
        s.recall = correctMatches / (float) visibleFeatures;//������������ͼ���ظ���������ȷƥ�����

		s.correctMatchesPercent = s.precision;
		s.percentOfMatches = (float)matchesCount / (s.totalKeypoints);
		//matchingRatio��ͨ�����϶��ߵĳ˻�����

        
        // Compute matching statistics �ⲿ��һ��ʼҲ��ע�͵���
        if (homographyFound)
        {
            cv::Mat r = expectedHomography * homography.inv();//��������
            float error = cv::norm(cv::Mat::eye(3,3, CV_64FC1) - r, cv::NORM_INF);//��������� �õ������������Ϊ��λ norm������ ����Ԫ�ؾ���ֵ��ƽ�����ٿ�ƽ��
			// ֪���� ʹ��opencv��ν������������㼯֮������ƶȱȽϵģ���һ���𰸾����÷���

            computeMatchesDistanceStatistics(inliermatches, s.meanDistance, s.stdDevDistance);//������Զ�֮��ľ����ֵ�ͱ�׼�� ���������ƥ�����RANSAC֮ǰ����֮���أ�
            s.reprojectionError = computeReprojectionError(sourceKp, resKpReal, matches, homography);//�õ���ԶԵľ����ֵ����׼�� ���ֵ ��Сֵ��ɵ�scalar
            s.homographyError = std::min(error, 1.0f);//ֱ��1.0��double�ͣ������f���float�͵�
			//ֻ���С��1�Ĵ���
			
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
                //��������û��int�͵�correctmatches http://blog.sina.com.cn/sblog_a98e39a201017pgn.html
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
