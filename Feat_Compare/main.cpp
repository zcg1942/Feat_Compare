
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <algorithm>
#include <numeric>
#include <fstream>


const bool USE_VERBOSE_TRANSFORMATIONS = false;//默认不显示详细进度，所以step比较大

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useBF = true;//useBruteForceMather

    // Initialize list of algorithm tuples:

    algorithms.push_back(FeatureAlgorithm("ORB",   cv::Feature2D::create("ORB"),   useBF));//第二个参数，算法类型就是featureEngine
	algorithms.push_back(FeatureAlgorithm("SIFT", cv::Feature2D::create("SIFT"), useBF));
	algorithms.push_back(FeatureAlgorithm("BRISK", cv::Feature2D::create("BRISK"), useBF));
	algorithms.push_back(FeatureAlgorithm("SURF", cv::Feature2D::create("SURF"), useBF));
	algorithms.push_back(FeatureAlgorithm("FREAK", cv::Ptr<cv::FeatureDetector>(new cv::SurfFeatureDetector(2000, 4)), cv::Ptr<cv::DescriptorExtractor>(new cv::FREAK()), useBF));
    algorithms.push_back(FeatureAlgorithm("AKAZE", cv::Feature2D::create("AKAZE"), useBF));
    //algorithms.push_back(FeatureAlgorithm("KAZE",  cv::Feature2D::create("KAZE"),  useBF));//KAZE没有添加到Feature2D中
	
	

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS)
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 1)));//亮度从-127~127，步长为1
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f, 0.5f))));//步长为1
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

        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)//遍历每种算法
        {
            const FeatureAlgorithm& alg   = algorithms[algIndex];

            std::cout << "Testing " << alg.name << "...";

			//对测试图进行各种变换
            for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
            {
                const ImageTransformation& trans = *transformations[transformIndex];

                performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));//返回一个bool类型，在这个过程中完成图像配准 很重要！！！！在AloorithmEstimation中
				//getStatistics函数返回算法名字和变换类型名字组成的map
				//这个函数是AlgorithmEstimation.cpp中最重要的函数
            }

            std::cout << "done." << std::endl;
        }

		//将测试的平均值打印在控制台中
        fullStat.printAverage(std::cout, StatisticsElementRecall);
        fullStat.printAverage(std::cout, StatisticsElementPrecision);

		fullStat.printAverage(std::cout, StatisticsElementHomographyError);
		fullStat.printAverage(std::cout, StatisticsElementMatchingRatio);
		fullStat.printAverage(std::cout, StatisticsElementMeanDistance);
		fullStat.printAverage(std::cout, StatisticsElementPercentOfCorrectMatches);
		fullStat.printAverage(std::cout, StatisticsElementPercentOfMatches);
		fullStat.printAverage(std::cout, StatisticsElementPointsCount);

		fullStat.printPerformanceStatistics(std::cout);

	    //将测试的各个值保存在txt中，再用matlab绘图
        std::ofstream recallLog("Recall.txt");//文件写操作并关联文件 内存写入存储设备   
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
	getchar();//为了使控制台不闪退
    return 0;
}

