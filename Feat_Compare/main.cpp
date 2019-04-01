
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <algorithm>
#include <numeric>
#include <fstream>


const bool USE_VERBOSE_TRANSFORMATIONS = false;//Ĭ�ϲ���ʾ��ϸ���ȣ�����step�Ƚϴ�

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useBF = true;//useBruteForceMather

    // Initialize list of algorithm tuples:

    algorithms.push_back(FeatureAlgorithm("ORB",   cv::Feature2D::create("ORB"),   useBF));//�ڶ����������㷨���;���featureEngine
	algorithms.push_back(FeatureAlgorithm("SIFT", cv::Feature2D::create("SIFT"), useBF));
	algorithms.push_back(FeatureAlgorithm("BRISK", cv::Feature2D::create("BRISK"), useBF));
	algorithms.push_back(FeatureAlgorithm("SURF", cv::Feature2D::create("SURF"), useBF));
	algorithms.push_back(FeatureAlgorithm("FREAK", cv::Ptr<cv::FeatureDetector>(new cv::SurfFeatureDetector(2000, 4)), cv::Ptr<cv::DescriptorExtractor>(new cv::FREAK()), useBF));
    algorithms.push_back(FeatureAlgorithm("AKAZE", cv::Feature2D::create("AKAZE"), useBF));
    //algorithms.push_back(FeatureAlgorithm("KAZE",  cv::Feature2D::create("KAZE"),  useBF));//KAZEû����ӵ�Feature2D��
	
	

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS)
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 1)));//���ȴ�-127~127������Ϊ1
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f, 0.5f))));//����Ϊ1
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.01f)));//0.01f step
    }
    else
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 10, cv::Point2f(0.5f, 0.5f))));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.1f)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 10)));
    }

    if (argc < 2)
    {
        std::cout << "At least one input image should be passed" << std::endl;
    }

    for (int imageIndex = 1; imageIndex < argc; imageIndex++)
    {
        std::string testImagePath(argv[imageIndex]);
        cv::Mat testImage = cv::imread(testImagePath);

        CollectedStatistics fullStat;

        if (testImage.empty())
        {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }

        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)//����ÿ���㷨
        {
            const FeatureAlgorithm& alg   = algorithms[algIndex];

            std::cout << "Testing " << alg.name << "...";

			//�Բ���ͼ���и��ֱ任
            for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
            {
                const ImageTransformation& trans = *transformations[transformIndex];

                performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));//����һ��bool���ͣ���������������ͼ����׼ ����Ҫ����������AloorithmEstimation��
				//getStatistics���������㷨���ֺͱ任����������ɵ�map
				//���������AlgorithmEstimation.cpp������Ҫ�ĺ���
            }

            std::cout << "done." << std::endl;
        }

		//�����Ե�ƽ��ֵ��ӡ�ڿ���̨��
        fullStat.printAverage(std::cout, StatisticsElementRecall);
        fullStat.printAverage(std::cout, StatisticsElementPrecision);

		fullStat.printAverage(std::cout, StatisticsElementHomographyError);
		fullStat.printAverage(std::cout, StatisticsElementMatchingRatio);
		fullStat.printAverage(std::cout, StatisticsElementMeanDistance);
		fullStat.printAverage(std::cout, StatisticsElementPercentOfCorrectMatches);
		fullStat.printAverage(std::cout, StatisticsElementPercentOfMatches);
		fullStat.printAverage(std::cout, StatisticsElementPointsCount);

		fullStat.printPerformanceStatistics(std::cout);

	    //�����Եĸ���ֵ������txt�У�����matlab��ͼ
        std::ofstream recallLog("Recall.txt");//�ļ�д�����������ļ� �ڴ�д��洢�豸   
        fullStat.printStatistics(recallLog, StatisticsElementRecall);
        std::ofstream precisionLog("Precision.txt");
        fullStat.printStatistics(precisionLog, StatisticsElementPrecision);

		std::ofstream HomographyErrorLog(" HomographyError.txt ");
		fullStat.printStatistics(HomographyErrorLog, StatisticsElementHomographyError);
		std::ofstream MatchingRatioLog("MatchingRatio.txt");
		fullStat.printStatistics(MatchingRatioLog, StatisticsElementMatchingRatio);
		std::ofstream MeanDistanceLog("MeanDistance.txt");
		fullStat.printStatistics(MeanDistanceLog, StatisticsElementMeanDistance);
		std::ofstream PercentOfCorrectMatchesLog("PercentOfCorrectMatches.txt  ");
		fullStat.printStatistics(PercentOfCorrectMatchesLog, StatisticsElementPercentOfCorrectMatches);
		std::ofstream PercentOfMatchesLog("PercentOfMatches.txt");
		fullStat.printStatistics(PercentOfMatchesLog, StatisticsElementPercentOfMatches);
		std::ofstream PerformanceLog("Performance.txt");
		fullStat.printPerformanceStatistics(PerformanceLog); 

    }
	getchar();//Ϊ��ʹ����̨������
    return 0;
}

