/************************************************************************************\
    This is improved variant of chessboard corner detection algorithm that
    uses a graph of connected quads. It is based on the code contributed
    by Vladimir Vezhnevets and Philip Gruebele.
    Here is the copyright notice from the original Vladimir's code:
    ===============================================================

    The algorithms developed and implemented by Vezhnevets Vldimir
    aka Dead Moroz (vvp@graphics.cs.msu.ru)
    See http://graphics.cs.msu.su/en/research/calibration/opencv.html
    for detailed information.

    Reliability additions and modifications made by Philip Gruebele.
    <a href="mailto:pgruebele@cox.net">pgruebele@cox.net</a>

	His code was adapted for use with low resolution and omnidirectional cameras
	by Martin Rufli during his Master Thesis under supervision of Davide Scaramuzza, at the ETH Zurich. Further enhancements include:
		- Increased chance of correct corner matching.
		- Corner matching over all dilation runs.
		
If you use this code, please cite the following articles:

1. Scaramuzza, D., Martinelli, A. and Siegwart, R. (2006), A Toolbox for Easily Calibrating Omnidirectional Cameras, Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems  (IROS 2006), Beijing, China, October 2006.
2. Scaramuzza, D., Martinelli, A. and Siegwart, R., (2006). "A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion", Proceedings of IEEE International Conference of Vision Systems  (ICVS'06), New York, January 5-7, 2006.
3. Rufli, M., Scaramuzza, D., and Siegwart, R. (2008), Automatic Detection of Checkerboards on Blurred and Distorted Images, Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2008), Nice, France, September 2008.

\************************************************************************************/

#define index2d(X, Y) (m_imCurr->cols * (Y) + (X))

//===========================================================================
// CODE STARTS HERE
//===========================================================================
// Include files
#include <opencv.hpp>
extern "C" {
#include <opencv2/core/core_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>

}

#include <time.h>
#include <fstream>
#include <csignal>
using namespace std;
using std::ifstream;


// Defines
#define MAX_CONTOUR_APPROX  7



//Ming #define VIS 1
#define VIS 1
// Turn on visualization
#define TIMER 0					// Elapse the function duration times


// Definition Contour Struct
typedef struct CvContourEx
{
    CV_CONTOUR_FIELDS()
    int counter;
}
CvContourEx;


// Definition Corner Struct
typedef struct CvCBCorner
{
    CvPoint2D32f pt;					// X and y coordinates
	int row;							// Row and column of the corner 
	int column;							// in the found pattern
	bool needsNeighbor;					// Does the corner require a neighbor?
    int count;							// number of corner neighbors
    struct CvCBCorner* neighbors[4];	// pointer to all corner neighbors
}
CvCBCorner;


// Definition Quadrangle Struct
// This structure stores information about the chessboard quadrange
typedef struct CvCBQuad
{
    int count;							// Number of quad neihbors
    int group_idx;						// Quad group ID
    float edge_len;						// Smallest side length^2
    CvCBCorner *corners[4];				// Coordinates of quad corners
    struct CvCBQuad *neighbors[4];		// Pointers of quad neighbors
	bool labeled;						// Has this corner been labeled?
}
CvCBQuad;



typedef struct GridPoint{
  cv::Point* point;
  GridPoint* left;
  GridPoint* right;
  GridPoint* above;
  GridPoint* below;;
  GridPoint(cv::Point* i_point){
    point = i_point;
    left = right = above = below = nullptr;
  };
}
GridPoint;

typedef struct HullPoint{
  cv::Point* point;
  HullPoint* left;
  HullPoint* right;
  double angle = -1;
  HullPoint(cv::Point* i_point){
    point = i_point;
    left = right = nullptr;
  };
}
HullPoint;

static bool DEBUG;

//===========================================================================
// FUNCTION PROTOTYPES
//===========================================================================
static int icvGenerateQuads( CvCBQuad **quads, CvCBCorner **corners,
                             CvMemStorage *storage, CvMat *image, int flags, int dilation,
							 bool firstRun );

static void mrFindQuadNeighbors2( CvCBQuad *quads, int quad_count, int dilation);

static int mrAugmentBestRun( CvCBQuad *new_quads, int new_quad_count, int new_dilation, 
							 CvCBQuad **old_quads, int old_quad_count, int old_dilation );

static int icvFindConnectedQuads( CvCBQuad *quads, int quad_count, CvCBQuad **quad_group,
								  int group_idx,
                                  CvMemStorage* storage, int dilation );

static void mrLabelQuadGroup( CvCBQuad **quad_group, int count, CvSize pattern_size, 
							  bool firstRun );

static void mrCopyQuadGroup( CvCBQuad **temp_quad_group, CvCBQuad **out_quad_group, 
							 int count );

static int icvCleanFoundConnectedQuads( int quad_count, CvCBQuad **quads, 
									    CvSize pattern_size );

static int mrWriteCorners( CvCBQuad **output_quads, int count, CvSize pattern_size,
						   int min_number_of_corners );

static void initFAST(std::vector<cv::Point> &fastPoints, std::vector<cv::Point> &fastInterior);

static void getUVPoints(const CvMat *i_imCurr, std::vector<cv::Point2i> &outvec, int threshVal);

static void getHullPoints(std::vector<cv::Point2i> &uv_points,  HullPoint*& start, double maxConcaveAngle, double similarAngle, CvMat* vis_img);

static HullPoint* getSharpestHullPoint(HullPoint* start);

static void getGrid(HullPoint* corner_hull_point, GridPoint*& grid_points, bool& x_axis_first, std::vector<cv::Point2i> &uv_points,  int W, int H, CvMat *vis_img);

static int mrWriteMarkers(GridPoint*& grid_points, bool x_axis_first, CvSize pattern_size, int min_number_of_corners );

//===========================================================================
// MAIN FUNCTION
//===========================================================================
int cvFindUVMarkers( const void* arr, CvSize pattern_size,
                             CvPoint2D32f* out_corners, int* out_corner_count,
                             int min_number_of_corners, bool set_debug )
{
  DEBUG = set_debug;
//START TIMER
#if TIMER
	ofstream FindUVMarkersStream;
    time_t  start_time = clock();
#endif

	// PART 0: INITIALIZATION
	//-----------------------------------------------------------------------
	// Initialize variables
	int flags					=  1;	// not part of the function call anymore!
	int max_count				=  0;
	int max_dilation_run_ID		= -1;
    const int min_dilations		=  1;
    const int max_dilations		=  6;
    int found					=  0;
    CvMat* norm_img				=  0;
    CvMat* thresh_img			=  0;
	CvMat* thresh_img_save		=  0;
    CvMemStorage* storage		=  0;
#if VIS
    CvMat* vis_img			=  0;
#endif
	
	CvCBQuad *quads				=  0;
	CvCBQuad **quad_group		=  0;
    CvCBCorner *corners			=  0;
	CvCBCorner **corner_group	=  0;
	CvCBQuad **output_quad_group = 0;
  std::vector<cv::Point2i> uv_points = std::vector<cv::Point2i>();
  std::vector<bool> marked_points = std::vector<bool>();
  HullPoint* hull_points;
  HullPoint* corner_hull_point;
  HullPoint * currHullPoint = hull_points;
  GridPoint* grid_points;
  bool x_axis_first;
	
    // debug trial. Martin Rufli, 28. Ocober, 2008
	int block_size = 0;


	// Create error message file
	ofstream error("cToMatlab/error.txt");

	
	// Set openCV function name and label the function start
    CV_FUNCNAME( "cvFindUVMarkers" );
    __CV_BEGIN__;


	// Further initializations
    int quad_count, group_idx, dilations;
    CvMat stub, *img = (CvMat*)arr;


	// Read image from input
    CV_CALL( img = cvGetMat( img, &stub ));


	// Error handling, write error message to error.txt
    if( CV_MAT_DEPTH( img->type ) != CV_8U || CV_MAT_CN( img->type ) == 2 )
	{
        error << "Only 8-bit grayscale or color images are supported" << endl;
		error.close();
		return -1;
	}
    if( pattern_size.width < 2 || pattern_size.height < 2 )
	{
        error << "Pattern should have at least 2x2 size" << endl;
		error.close();
		return -1;
	}
	if( pattern_size.width > 127 || pattern_size.height > 127 )
	{
        error << "Pattern should not have a size larger than 127 x 127" << endl;
		error.close();
		return -1;
	}
	/*
	if( pattern_size.width != pattern_size.height )
	{
        error << "In this implementation only square sized checker boards are supported" << endl;
		error.close();
		return -1;
	}
	*/
	if( !out_corners )
	{
        error << "Null pointer to corners encountered" << endl;
		error.close();
		return -1;
	}


	// Create memory storage
    CV_CALL( storage = cvCreateMemStorage(0) );
    CV_CALL( thresh_img = cvCreateMat( img->rows, img->cols, CV_8UC1 ));
	CV_CALL( thresh_img_save = cvCreateMat( img->rows, img->cols, CV_8UC1 ));


	// Image histogramm normalization and
	// BGR to Grayscale image conversion (if applicable)
	// MARTIN: Set to "false"
    if( CV_MAT_CN(img->type) != 1 || (flags & CV_CALIB_CB_NORMALIZE_IMAGE) )
    {
        CV_CALL( norm_img = cvCreateMat( img->rows, img->cols, CV_8UC1 ));

        if( CV_MAT_CN(img->type) != 1 )
        {
            CV_CALL( cvCvtColor( img, norm_img, CV_BGR2GRAY ));
            img = norm_img;
        }

        if(false)
        {
            cvEqualizeHist( img, norm_img );
            img = norm_img;
        }
    }
	
// EVALUATE TIMER
#if TIMER
	float time0_1 = (float) (clock() - start_time) / CLOCKS_PER_SEC;
	FindUVMarkersStream.open("timer/FindUVMarkers.txt", ofstream::app);
	FindUVMarkersStream << "Time 0.1 for cvFindChessboardCorners2 was " << time0_1 << " seconds." << endl;
#endif


//V: START EDITING HERE


  CV_CALL( getUVPoints(img, uv_points, 150) );
  for (auto &uvpt : uv_points)
    marked_points.push_back(false);
//VISUALIZATION--------------------------------------------------------------
#if VIS
    CV_CALL( vis_img = cvCreateMat( img->rows, img->cols, CV_8UC1 ));
    cvCopy( img, vis_img);
    for (auto &pt : uv_points){
      /* std::cout << pt << std::endl; */
      cvCircle(vis_img, pt, 5, cv::Scalar(255));
    }
 		cvNamedWindow( "ocv_Markers", 1 );
		cvShowImage( "ocv_Markers", vis_img);
		//cvSaveImage("pictureVis/OrigImg.png", img);
		cvWaitKey(10);
#endif
//END------------------------------------------------------------------------
//
  CV_CALL( getHullPoints(uv_points, hull_points, M_PI/4, M_PI/9, vis_img) );

  CV_CALL( corner_hull_point = getSharpestHullPoint(hull_points));

//VISUALIZATION--------------------------------------------------------------
#if VIS
  cvCircle(vis_img, *(corner_hull_point->point), 15, cv::Scalar(150));
  cvShowImage( "ocv_Markers", vis_img);
  //cvSaveImage("pictureVis/OrigImg.png", img);
		cvWaitKey(10);
#endif
//END------------------------------------------------------------------------
//
  CV_CALL( getGrid(corner_hull_point, grid_points, x_axis_first, uv_points, pattern_size.width, pattern_size.height, vis_img));

  CV_CALL( found = mrWriteMarkers(grid_points, x_axis_first, pattern_size, pattern_size.width*pattern_size.height));
  error.close();

        
  if (found == -1 || found == 1)
    __CV_EXIT__;

	// "End of file" jump point
	// After the command "__CV_EXIT__" the code jumps here
    __CV_END__;


	/*
	// MARTIN:
	found = mrWriteCorners( output_quad_group, max_count, pattern_size, min_number_of_corners);
	*/

	// If a linking problem was encountered, throw an error message
    if( found == -1 )
	{
        error << "While linking the corners a problem was encountered. No corner sequence is returned. " << endl;
		error.close();
		return -1;
	}


	// Release allocated memory
    cvReleaseMemStorage( &storage );
    cvReleaseMat( &norm_img );
    cvReleaseMat( &thresh_img );
    cvFree( &quads );
    cvFree( &corners );
    cvFree( &quad_group );
    cvFree( &corner_group );
	cvFree( &output_quad_group );
	error.close();

// EVALUATE TIMER
#if TIMER
	float time3 = (float) (clock() - start_time) / CLOCKS_PER_SEC;
	FindUVMarkersStream << "Time 3 for cvFindChessboardCorners2 was " << time3 << " seconds." << endl;
	FindUVMarkersStream.close();
#endif

	// Return found
	// Found can have the values
	// -1  ->	Error or corner linking problem, see error.txt for more information
	//  0  ->	Not enough corners were found
	//  1  ->	Enough corners were found
    return found;
}


//===========================================================================
// ERASE OVERHEAD
//===========================================================================
// If we found too many connected quads, remove those which probably do not 
// belong.
static int
icvCleanFoundConnectedQuads( int quad_count, CvCBQuad **quad_group, CvSize pattern_size )
{
    CvMemStorage *temp_storage = 0;
    CvPoint2D32f *centers = 0;

    CV_FUNCNAME( "icvCleanFoundConnectedQuads" );

    __CV_BEGIN__;

    CvPoint2D32f center = {0,0};
    int i, j, k;


    // Number of quads this pattern should contain
    int count = ((pattern_size.width + 1)*(pattern_size.height + 1) + 1)/2;


    // If we have more quadrangles than we should, try to eliminate duplicates
	// or ones which don't belong to the pattern rectangle. Else go to the end
	// of the function
    if( quad_count <= count )
        __CV_EXIT__;


    // Create an array of quadrangle centers
    CV_CALL( centers = (CvPoint2D32f *)cvAlloc( sizeof(centers[0])*quad_count ));
    CV_CALL( temp_storage = cvCreateMemStorage(0));

    for( i = 0; i < quad_count; i++ )
    {
        CvPoint2D32f ci = {0,0};
        CvCBQuad* q = quad_group[i];

        for( j = 0; j < 4; j++ )
        {
            CvPoint2D32f pt = q->corners[j]->pt;
            ci.x += pt.x;
            ci.y += pt.y;
        }

        ci.x *= 0.25f;
        ci.y *= 0.25f;
	

		// Centers(i), is the geometric center of quad(i)
		// Center, is the center of all found quads
        centers[i] = ci;
        center.x += ci.x;
        center.y += ci.y;
    }
    center.x /= quad_count;
    center.y /= quad_count;

    // If we have more quadrangles than we should, we try to eliminate bad
	// ones based on minimizing the bounding box. We iteratively remove the
	// point which reduces the size of the bounding box of the blobs the most
    // (since we want the rectangle to be as small as possible) remove the
	// quadrange that causes the biggest reduction in pattern size until we
	// have the correct number
    for( ; quad_count > count; quad_count-- )
    {
        double min_box_area = DBL_MAX;
        int skip, min_box_area_index = -1;
        CvCBQuad *q0, *q;


        // For each point, calculate box area without that point
        for( skip = 0; skip < quad_count; skip++ )
        {
            // get bounding rectangle
            CvPoint2D32f temp = centers[skip]; 
            centers[skip] = center; 
            CvMat pointMat = cvMat(1, quad_count, CV_32FC2, centers);
            CvSeq *hull = cvConvexHull2( &pointMat, temp_storage, CV_CLOCKWISE, 1 );
            centers[skip] = temp;
            double hull_area = fabs(cvContourArea(hull, CV_WHOLE_SEQ));


            // remember smallest box area
            if( hull_area < min_box_area )
            {
                min_box_area = hull_area;
                min_box_area_index = skip;
            }
            cvClearMemStorage( temp_storage );
        }

        q0 = quad_group[min_box_area_index];


        // remove any references to this quad as a neighbor
        for( i = 0; i < quad_count; i++ )
        {
            q = quad_group[i];
            for( j = 0; j < 4; j++ )
            {
                if( q->neighbors[j] == q0 )
                {
                    q->neighbors[j] = 0;
                    q->count--;
                    for( k = 0; k < 4; k++ )
                        if( q0->neighbors[k] == q )
                        {
                            q0->neighbors[k] = 0;
                            q0->count--;
                            break;
                        }
                    break;
                }
            }
        }

		// remove the quad by copying th last quad in the list into its place
        quad_count--;
        quad_group[min_box_area_index] = quad_group[quad_count];
        centers[min_box_area_index] = centers[quad_count];
    }

    __CV_END__;

    cvReleaseMemStorage( &temp_storage );
    cvFree( &centers );

    return quad_count;
}



//===========================================================================
// FIND COONECTED QUADS
//===========================================================================
static int
icvFindConnectedQuads( CvCBQuad *quad, int quad_count, CvCBQuad **out_group,
                       int group_idx, CvMemStorage* storage, int dilation )
{
//START TIMER
#if TIMER
	ofstream FindConnectedQuads;
    time_t  start_time = clock();
#endif

	// initializations
    CvMemStorage* temp_storage = cvCreateChildMemStorage( storage );
    CvSeq* stack = cvCreateSeq( 0, sizeof(*stack), sizeof(void*), temp_storage );
	int i, count = 0;


    // Scan the array for a first unlabeled quad
    for( i = 0; i < quad_count; i++ )
    {
        if( quad[i].count > 0 && quad[i].group_idx < 0)
            break;
    }


    // Recursively find a group of connected quads starting from the seed
	// quad[i]
    if( i < quad_count )
    {
        CvCBQuad* q = &quad[i];
        cvSeqPush( stack, &q );
        out_group[count++] = q;
        q->group_idx = group_idx;

        while( stack->total )
        {
            cvSeqPop( stack, &q );
            for( i = 0; i < 4; i++ )
            {
                CvCBQuad *neighbor = q->neighbors[i];


				// If he neighbor exists and the neighbor has more than 0 
				// neighbors and the neighbor has not been classified yet.
                if( neighbor && neighbor->count > 0 && neighbor->group_idx < 0 )
                {
                    cvSeqPush( stack, &neighbor );
                    out_group[count++] = neighbor;
                    neighbor->group_idx = group_idx;
                }
            }
        }
    }

    cvReleaseMemStorage( &temp_storage );
	
// EVALUATE TIMER
#if TIMER
	float time = (float) (clock() - start_time) / CLOCKS_PER_SEC;
	FindConnectedQuads.open("timer/FindConnectedQuads.txt", ofstream::app);
	FindConnectedQuads << "Time for cvFindConnectedQuads was " << time << " seconds." << endl;
	FindConnectedQuads.close();
#endif

    return count;
}



//===========================================================================
// LABEL CORNER WITH ROW AND COLUMN //DONE
//===========================================================================
static void mrLabelQuadGroup( CvCBQuad **quad_group, int count, CvSize pattern_size, bool firstRun )
{
//START TIMER
#if TIMER
	ofstream LabelQuadGroup;
    time_t  start_time = clock();
#endif

	// If this is the first function call, a seed quad needs to be selected
	if (firstRun == true)
	{
		// Search for the (first) quad with the maximum number of neighbors
		// (usually 4). This will be our starting point.
		int max_id = -1;
		int max_number = -1;
		for(int i = 0; i < count; i++ )
		{
			CvCBQuad* q = quad_group[i];
			if( q->count > max_number)
			{
				max_number = q->count;
				max_id = i;

				if (max_number == 4)
					break;
			}
		}


		// Mark the starting quad's (per definition) upper left corner with
		//(0,0) and then proceed clockwise
		// The following labeling sequence ensures a "right coordinate system"
		(quad_group[max_id])->labeled = true;

		(quad_group[max_id])->corners[0]->row = 0;
        (quad_group[max_id])->corners[0]->column = 0;
		(quad_group[max_id])->corners[1]->row = 0;
		(quad_group[max_id])->corners[1]->column = 1;
		(quad_group[max_id])->corners[2]->row = 1;
		(quad_group[max_id])->corners[2]->column = 1;
		(quad_group[max_id])->corners[3]->row = 1;
		(quad_group[max_id])->corners[3]->column = 0;
	}


	// Mark all other corners with their respective row and column
	bool flag_changed = true;
	while( flag_changed == true )
	{
		// First reset the flag to "false"
		flag_changed = false;


		// Going through all quads top down is faster, since unlabeled quads will
		// be inserted at the end of the list
		for( int i = (count-1); i >= 0; i-- )
		{
			// Check whether quad "i" has been labeled already
 			if ( (quad_group[i])->labeled == false )
			{
				// Check its neighbors, whether some of them have been labeled
				// already
				for( int j = 0; j < 4; j++ )
				{
					// Check whether the neighbor exists (i.e. is not the NULL
					// pointer)
					if( (quad_group[i])->neighbors[j] )
					{
						CvCBQuad *quadNeighborJ = (quad_group[i])->neighbors[j];
						
						
						// Only proceed, if neighbor "j" was labeled
						if( quadNeighborJ->labeled == true)
						{
							// For every quad it could happen to pass here 
							// multiple times. We therefore "break" later.
							// Check whitch of the neighbors corners is 
							// connected to the current quad
							int connectedNeighborCornerId = -1;
							for( int k = 0; k < 4; k++)
							{
								if( quadNeighborJ->neighbors[k] == quad_group[i] )
								{
									connectedNeighborCornerId = k;
									
									
									// there is only one, therefore
									break;
								}
							}


							// For the following calculations we need the row 
							// and column of the connected neighbor corner and 
							// all other corners of the connected quad "j", 
							// clockwise (CW)
							CvCBCorner *conCorner	 = quadNeighborJ->corners[connectedNeighborCornerId];
							CvCBCorner *conCornerCW1 = quadNeighborJ->corners[(connectedNeighborCornerId+1)%4];
							CvCBCorner *conCornerCW2 = quadNeighborJ->corners[(connectedNeighborCornerId+2)%4];
							CvCBCorner *conCornerCW3 = quadNeighborJ->corners[(connectedNeighborCornerId+3)%4];
							
							(quad_group[i])->corners[j]->row			=	conCorner->row;
							(quad_group[i])->corners[j]->column			=	conCorner->column;
							(quad_group[i])->corners[(j+1)%4]->row		=	conCorner->row - conCornerCW2->row + conCornerCW3->row;
							(quad_group[i])->corners[(j+1)%4]->column	=	conCorner->column - conCornerCW2->column + conCornerCW3->column;
							(quad_group[i])->corners[(j+2)%4]->row		=	conCorner->row + conCorner->row - conCornerCW2->row;
							(quad_group[i])->corners[(j+2)%4]->column	=	conCorner->column + conCorner->column - conCornerCW2->column;
							(quad_group[i])->corners[(j+3)%4]->row		=	conCorner->row - conCornerCW2->row + conCornerCW1->row;
							(quad_group[i])->corners[(j+3)%4]->column	=	conCorner->column - conCornerCW2->column + conCornerCW1->column;
							

							// Mark this quad as labeled
							(quad_group[i])->labeled = true;
							

							// Changes have taken place, set the flag
							flag_changed = true;


							// once is enough!
							break;
						}
					}
				}
			}
		}
	}


	// All corners are marked with row and column
	// Record the minimal and maximal row and column indices
	// It is unlikely that more than 8bit checkers are used per dimension, if there are
	// an error would have been thrown at the beginning of "cvFindChessboardCorners2"
	int min_row		=  127;
	int max_row		= -127;
	int min_column	=  127;
	int max_column	= -127;

	for(int i = 0; i < count; i++ )
    {
		CvCBQuad* q = quad_group[i];
		
		for(int j = 0; j < 4; j++ )
		{
			if( (q->corners[j])->row > max_row)
				max_row = (q->corners[j])->row;

			if( (q->corners[j])->row < min_row)
				min_row = (q->corners[j])->row;

			if( (q->corners[j])->column > max_column)
				max_column = (q->corners[j])->column;

			if( (q->corners[j])->column < min_column)
				min_column = (q->corners[j])->column;
		}
	}

	// Label all internal corners with "needsNeighbor" = false
	// Label all external corners with "needsNeighbor" = true,
	// except if in a given dimension the pattern size is reached
	for(int i = min_row; i <= max_row; i++)
	{
			for(int j = min_column; j <= max_column; j++)
			{
				// A flag that indicates, wheter a row/column combination is
				// executed multiple times
				bool flagg = false;


				// Remember corner and quad
				int cornerID;
				int quadID;

				for(int k = 0; k < count; k++)
				{
					for(int l = 0; l < 4; l++)
					{
						if( ((quad_group[k])->corners[l]->row == i) && ((quad_group[k])->corners[l]->column == j) )
						{
							
							if (flagg == true)
							{
								// Passed at least twice through here
								(quad_group[k])->corners[l]->needsNeighbor = false;
								(quad_group[quadID])->corners[cornerID]->needsNeighbor = false;
							}
							else
							{
								// Mark with needs a neighbor, but note the
								// address
								(quad_group[k])->corners[l]->needsNeighbor = true;
								cornerID = l;
								quadID = k;
							}
							

							// set the flag to true
							flagg = true;
						}
					}
				}
			}
	}

	
	// Complete Linking:
	// sometimes not all corners were properly linked in "mrFindQuadNeighbors2",
	// but after labeling each corner with its respective row and column, it is 
	// possible to match them anyway.
	for(int i = min_row; i <= max_row; i++)
	{
			for(int j = min_column; j <= max_column; j++)
			{
				// the following "number" indicates the number of corners which 
				// correspond to the given (i,j) value
				// 1	is a border corner or a conrer which still needs a neighbor
				// 2	is a fully connected internal corner
				// >2	something went wrong during labeling, report a warning
				int number = 1;


				// remember corner and quad
				int cornerID;
				int quadID;

				for(int k = 0; k < count; k++)
				{
					for(int l = 0; l < 4; l++)
					{
						if( ((quad_group[k])->corners[l]->row == i) && ((quad_group[k])->corners[l]->column == j) )
						{

							if (number == 1)
							{
								// First corner, note its ID
								cornerID = l;
								quadID = k;
							}
							
							else if (number == 2)
							{
								// Second corner, check wheter this and the 
								// first one have equal coordinates, else 
								// interpolate
								float delta_x = (quad_group[k])->corners[l]->pt.x - (quad_group[quadID])->corners[cornerID]->pt.x;
								float delta_y = (quad_group[k])->corners[l]->pt.y - (quad_group[quadID])->corners[cornerID]->pt.y;
								
								if (delta_x != 0 || delta_y != 0)
								{
									// Interpolate
									(quad_group[k])->corners[l]->pt.x = (quad_group[k])->corners[l]->pt.x - delta_x/2;
									(quad_group[quadID])->corners[cornerID]->pt.x = (quad_group[quadID])->corners[cornerID]->pt.x + delta_x/2;
									(quad_group[k])->corners[l]->pt.y = (quad_group[k])->corners[l]->pt.y - delta_y/2;
									(quad_group[quadID])->corners[cornerID]->pt.y = (quad_group[quadID])->corners[cornerID]->pt.y + delta_y/2;
								}
							}
							else if (number > 2)
							{
								// Something went wrong during row/column labeling
								// Report a Warning
								// ->Implemented in the function "mrWriteCorners"
							}
	
							// increase the number by one
							number = number + 1;
						}
					}
				}
			}
	}


	// Bordercorners don't need any neighbors, if the pattern size in the 
	// respective direction is reached
	// The only time we can make sure that the target pattern size is reached in a given
	// dimension, is when the larger side has reached the target size in the maximal
	// direction, or if the larger side is larger than the smaller target size and the 
	// smaller side equals the smaller target size
	int largerDimPattern = max(pattern_size.height,pattern_size.width);
	int smallerDimPattern = min(pattern_size.height,pattern_size.width);
	bool flagSmallerDim1 = false;
	bool flagSmallerDim2 = false;

	if((largerDimPattern + 1) == max_column - min_column)
	{
		flagSmallerDim1 = true;
		// We found out that in the column direction the target pattern size is reached
		// Therefore border column corners do not need a neighbor anymore
		// Go through all corners
		for( int k = 0; k < count; k++ )
		{
			for( int l = 0; l < 4; l++ )
			{
				if ( (quad_group[k])->corners[l]->column == min_column || (quad_group[k])->corners[l]->column == max_column)
				{
					// Needs no neighbor anymore
					(quad_group[k])->corners[l]->needsNeighbor = false;
				}
			}
		}		
	}

	if((largerDimPattern + 1) == max_row - min_row)
	{
		flagSmallerDim2 = true;
		// We found out that in the column direction the target pattern size is reached
		// Therefore border column corners do not need a neighbor anymore
		// Go through all corners
		for( int k = 0; k < count; k++ )
		{
			for( int l = 0; l < 4; l++ )
			{
				if ( (quad_group[k])->corners[l]->row == min_row || (quad_group[k])->corners[l]->row == max_row)
				{
					// Needs no neighbor anymore
					(quad_group[k])->corners[l]->needsNeighbor = false;
				}
			}
		}		
	}


	// Check the two flags: 
	//	-	If one is true and the other false, then the pattern target 
	//		size was reached in in one direction -> We can check, whether the target 
	//		pattern size is also reached in the other direction
	//  -	If both are set to true, then we deal with a square board -> do nothing
	//  -	If both are set to false -> There is a possibility that the larger side is
	//		larger than the smaller target size -> Check and if true, then check whether
	//		the other side has the same size as the smaller target size
	if( (flagSmallerDim1 == false && flagSmallerDim2 == true) )
	{
		// Larger target pattern size is in row direction, check wheter smaller target
		// pattern size is reached in column direction
		if((smallerDimPattern + 1) == max_column - min_column)
		{
			for( int k = 0; k < count; k++ )
			{
				for( int l = 0; l < 4; l++ )
				{
					if ( (quad_group[k])->corners[l]->column == min_column || (quad_group[k])->corners[l]->column == max_column)
					{
						// Needs no neighbor anymore
						(quad_group[k])->corners[l]->needsNeighbor = false;
					}
				}
			}
		}
	}

	if( (flagSmallerDim1 == true && flagSmallerDim2 == false) )
	{
		// Larger target pattern size is in column direction, check wheter smaller target
		// pattern size is reached in row direction
		if((smallerDimPattern + 1) == max_row - min_row)
		{
			for( int k = 0; k < count; k++ )
			{
				for( int l = 0; l < 4; l++ )
				{
					if ( (quad_group[k])->corners[l]->row == min_row || (quad_group[k])->corners[l]->row == max_row)
					{
						// Needs no neighbor anymore
						(quad_group[k])->corners[l]->needsNeighbor = false;
					}
				}
			}
		}
	}

	if( (flagSmallerDim1 == false && flagSmallerDim2 == false) && smallerDimPattern + 1 < max_column - min_column )
	{
		// Larger target pattern size is in column direction, check wheter smaller target
		// pattern size is reached in row direction
		if((smallerDimPattern + 1) == max_row - min_row)
		{
			for( int k = 0; k < count; k++ )
			{
				for( int l = 0; l < 4; l++ )
				{
					if ( (quad_group[k])->corners[l]->row == min_row || (quad_group[k])->corners[l]->row == max_row)
					{
						// Needs no neighbor anymore
						(quad_group[k])->corners[l]->needsNeighbor = false;
					}
				}
			}
		}
	}

	if( (flagSmallerDim1 == false && flagSmallerDim2 == false) && smallerDimPattern + 1 < max_row - min_row )
	{
		// Larger target pattern size is in row direction, check wheter smaller target
		// pattern size is reached in column direction
		if((smallerDimPattern + 1) == max_column - min_column)
		{
			for( int k = 0; k < count; k++ )
			{
				for( int l = 0; l < 4; l++ )
				{
          if ( (quad_group[k])->corners[l]->column == min_column || (quad_group[k])->corners[l]->column == max_column)
          {
            // Needs no neighbor anymore
            (quad_group[k])->corners[l]->needsNeighbor = false;
          }
        }
      }
    }
  }



  // EVALUATE TIMER
#if TIMER
  float time = (float) (clock() - start_time) / CLOCKS_PER_SEC;
  LabelQuadGroup.open("timer/LabelQuadGroup.txt", ofstream::app);
  LabelQuadGroup << "Time for mrLabelQuadGroup was " << time << " seconds." << endl;
  LabelQuadGroup.close();
#endif

}



//===========================================================================
// PRESERVE LARGEST QUAD GROUP
//===========================================================================
// Copies all necessary information of every quad of the largest found group
// into a new Quad struct array. 
// This information is then again needed in PART 2 of the MAIN LOOP
static void mrCopyQuadGroup( CvCBQuad **temp_quad_group, CvCBQuad **for_out_quad_group, int count )
{
  for (int i = 0; i < count; i++)
  {
    for_out_quad_group[i]				= new CvCBQuad;
    for_out_quad_group[i]->count		= temp_quad_group[i]->count;
    for_out_quad_group[i]->edge_len		= temp_quad_group[i]->edge_len;
    for_out_quad_group[i]->group_idx	= temp_quad_group[i]->group_idx;
    for_out_quad_group[i]->labeled		= temp_quad_group[i]->labeled;

    for (int j = 0; j < 4; j++)
    {
      for_out_quad_group[i]->corners[j]					= new CvCBCorner;
      for_out_quad_group[i]->corners[j]->pt.x				= temp_quad_group[i]->corners[j]->pt.x;
      for_out_quad_group[i]->corners[j]->pt.y				= temp_quad_group[i]->corners[j]->pt.y;
      for_out_quad_group[i]->corners[j]->row				= temp_quad_group[i]->corners[j]->row;
      for_out_quad_group[i]->corners[j]->column			= temp_quad_group[i]->corners[j]->column;
      for_out_quad_group[i]->corners[j]->needsNeighbor	= temp_quad_group[i]->corners[j]->needsNeighbor;
    }
  }
}



//===========================================================================
// GIVE A GROUP IDX
//===========================================================================
// This function replaces mrFindQuadNeighbors, which in turn replaced
// icvFindQuadNeighbors
static void mrFindQuadNeighbors2( CvCBQuad *quads, int quad_count, int dilation)
{
  //START TIMER
#if TIMER
  ofstream FindQuadNeighbors2;
  time_t  start_time = clock();
#endif

  // Thresh dilation is used to counter the effect of dilation on the
  // distance between 2 neighboring corners. Since the distance below is 
  // computed as its square, we do here the same. Additionally, we take the
  // conservative assumption that dilation was performed using the 3x3 CROSS
  // kernel, which coresponds to the 4-neighborhood.
  const float thresh_dilation = (float)(2*dilation+3)*(2*dilation+3)*2;	// the "*2" is for the x and y component
  int idx, i, k, j;														// the "3" is for initial corner mismatch
  float dx, dy, dist;
  int cur_quad_group = -1;


  // Find quad neighbors
  for( idx = 0; idx < quad_count; idx++ )
  {
    CvCBQuad* cur_quad = &quads[idx];


    // Go through all quadrangles and label them in groups
    // For each corner of this quadrangle
    for( i = 0; i < 4; i++ )
    {
      CvPoint2D32f pt;
      float min_dist = FLT_MAX;
      int closest_corner_idx = -1;
      CvCBQuad *closest_quad = 0;
      CvCBCorner *closest_corner = 0;

      if( cur_quad->neighbors[i] )
        continue;

      pt = cur_quad->corners[i]->pt;


      // Find the closest corner in all other quadrangles
      for( k = 0; k < quad_count; k++ )
      {
        if( k == idx )
          continue;

        for( j = 0; j < 4; j++ )
        {
          // If it already has a neighbor
          if( quads[k].neighbors[j] )
            continue;

          dx = pt.x - quads[k].corners[j]->pt.x;
          dy = pt.y - quads[k].corners[j]->pt.y;
          dist = dx * dx + dy * dy;


          // The following "if" checks, whether "dist" is the
          // shortest so far and smaller than the smallest
          // edge length of the current and target quads
          if( dist < min_dist && 
              dist <= (cur_quad->edge_len + thresh_dilation) &&
              dist <= (quads[k].edge_len + thresh_dilation)    )
          {
            // First Check everything from the viewpoint of the current quad
            // compute midpoints of "parallel" quad sides 1
            float x1 = (cur_quad->corners[i]->pt.x + cur_quad->corners[(i+1)%4]->pt.x)/2;
            float y1 = (cur_quad->corners[i]->pt.y + cur_quad->corners[(i+1)%4]->pt.y)/2;				
            float x2 = (cur_quad->corners[(i+2)%4]->pt.x + cur_quad->corners[(i+3)%4]->pt.x)/2;
            float y2 = (cur_quad->corners[(i+2)%4]->pt.y + cur_quad->corners[(i+3)%4]->pt.y)/2;	
            // compute midpoints of "parallel" quad sides 2
            float x3 = (cur_quad->corners[i]->pt.x + cur_quad->corners[(i+3)%4]->pt.x)/2;
            float y3 = (cur_quad->corners[i]->pt.y + cur_quad->corners[(i+3)%4]->pt.y)/2;				
            float x4 = (cur_quad->corners[(i+1)%4]->pt.x + cur_quad->corners[(i+2)%4]->pt.x)/2;
            float y4 = (cur_quad->corners[(i+1)%4]->pt.y + cur_quad->corners[(i+2)%4]->pt.y)/2;	

            // MARTIN: Heuristic
            // For the corner "j" of quad "k" to be considered, 
            // it needs to be on the same side of the two lines as 
            // corner "i". This is given, if the cross product has 
            // the same sign for both computations below:
            float a1 = x1 - x2;
            float b1 = y1 - y2;
            // the current corner
            float c11 = cur_quad->corners[i]->pt.x - x2;
            float d11 = cur_quad->corners[i]->pt.y - y2;
            // the candidate corner
            float c12 = quads[k].corners[j]->pt.x - x2;
            float d12 = quads[k].corners[j]->pt.y - y2;
            float sign11 = a1*d11 - c11*b1;
            float sign12 = a1*d12 - c12*b1;

            float a2 = x3 - x4;
            float b2 = y3 - y4;
            // the current corner
            float c21 = cur_quad->corners[i]->pt.x - x4;
            float d21 = cur_quad->corners[i]->pt.y - y4;
            // the candidate corner
            float c22 = quads[k].corners[j]->pt.x - x4;
            float d22 = quads[k].corners[j]->pt.y - y4;
            float sign21 = a2*d21 - c21*b2;
            float sign22 = a2*d22 - c22*b2;


            // Then make sure that two border quads of the same row or
            // column don't link. Check from the current corner's view,
            // whether the corner diagonal from the candidate corner
            // is also on the same side of the two lines as the current
            // corner and the candidate corner.
            float c13 = quads[k].corners[(j+2)%4]->pt.x - x2;
            float d13 = quads[k].corners[(j+2)%4]->pt.y - y2;
            float c23 = quads[k].corners[(j+2)%4]->pt.x - x4;
            float d23 = quads[k].corners[(j+2)%4]->pt.y - y4;
            float sign13 = a1*d13 - c13*b1;
            float sign23 = a2*d23 - c23*b2;


            // Then check everything from the viewpoint of the candidate quad
            // compute midpoints of "parallel" quad sides 1
            float u1 = (quads[k].corners[j]->pt.x + quads[k].corners[(j+1)%4]->pt.x)/2;
            float v1 = (quads[k].corners[j]->pt.y + quads[k].corners[(j+1)%4]->pt.y)/2;				
            float u2 = (quads[k].corners[(j+2)%4]->pt.x + quads[k].corners[(j+3)%4]->pt.x)/2;
            float v2 = (quads[k].corners[(j+2)%4]->pt.y + quads[k].corners[(j+3)%4]->pt.y)/2;	
            // compute midpoints of "parallel" quad sides 2
            float u3 = (quads[k].corners[j]->pt.x + quads[k].corners[(j+3)%4]->pt.x)/2;
            float v3 = (quads[k].corners[j]->pt.y + quads[k].corners[(j+3)%4]->pt.y)/2;				
            float u4 = (quads[k].corners[(j+1)%4]->pt.x + quads[k].corners[(j+2)%4]->pt.x)/2;
            float v4 = (quads[k].corners[(j+1)%4]->pt.y + quads[k].corners[(j+2)%4]->pt.y)/2;	

            // MARTIN: Heuristic
            // for the corner "j" of quad "k" to be considered, it 
            // needs to be on the same side of the two lines as 
            // corner "i". This is again given, if the cross
            //product has the same sign for both computations below:
            float a3 = u1 - u2;
            float b3 = v1 - v2;
            // the current corner
            float c31 = cur_quad->corners[i]->pt.x - u2;
            float d31 = cur_quad->corners[i]->pt.y - v2;
            // the candidate corner
            float c32 = quads[k].corners[j]->pt.x - u2;
            float d32 = quads[k].corners[j]->pt.y - v2;
            float sign31 = a3*d31-c31*b3;
            float sign32 = a3*d32-c32*b3;

            float a4 = u3 - u4;
            float b4 = v3 - v4;
            // the current corner
            float c41 = cur_quad->corners[i]->pt.x - u4;
            float d41 = cur_quad->corners[i]->pt.y - v4;
            // the candidate corner
            float c42 = quads[k].corners[j]->pt.x - u4;
            float d42 = quads[k].corners[j]->pt.y - v4;
            float sign41 = a4*d41-c41*b4;
            float sign42 = a4*d42-c42*b4;


            // Then make sure that two border quads of the same row or
            // column don't link. Check from the candidate corner's view,
            // whether the corner diagonal from the current corner
            // is also on the same side of the two lines as the current
            // corner and the candidate corner.
            float c33 = cur_quad->corners[(i+2)%4]->pt.x - u2;
            float d33 = cur_quad->corners[(i+2)%4]->pt.y - v2;
            float c43 = cur_quad->corners[(i+2)%4]->pt.x - u4;
            float d43 = cur_quad->corners[(i+2)%4]->pt.y - v4;
            float sign33 = a3*d33-c33*b3;
            float sign43 = a4*d43-c43*b4;


            // Check whether conditions are fulfilled
            if ( ((sign11 < 0 && sign12 < 0) || (sign11 > 0 && sign12 > 0))  && 
							 ((sign21 < 0 && sign22 < 0) || (sign21 > 0 && sign22 > 0))  &&
							 ((sign31 < 0 && sign32 < 0) || (sign31 > 0 && sign32 > 0))  &&   
							 ((sign41 < 0 && sign42 < 0) || (sign41 > 0 && sign42 > 0))  &&
							 ((sign11 < 0 && sign13 < 0) || (sign11 > 0 && sign13 > 0))  &&   
							 ((sign21 < 0 && sign23 < 0) || (sign21 > 0 && sign23 > 0))  &&
							 ((sign31 < 0 && sign33 < 0) || (sign31 > 0 && sign33 > 0))  &&   
							 ((sign41 < 0 && sign43 < 0) || (sign41 > 0 && sign43 > 0))    )
						
						{
							closest_corner_idx = j;
							closest_quad = &quads[k];
							min_dist = dist;
						}
                    }
                }
            }

            // Have we found a matching corner point?
            if( closest_corner_idx >= 0 && min_dist < FLT_MAX )
            {
                closest_corner = closest_quad->corners[closest_corner_idx];


                // Make sure that the closest quad does not have the current
				// quad as neighbor already
                for( j = 0; j < 4; j++ )
                {
                    if( closest_quad->neighbors[j] == cur_quad )
                        break;
                }
                if( j < 4 )
                    continue;


				// We've found one more corner - remember it
                closest_corner->pt.x = (pt.x + closest_corner->pt.x) * 0.5f;
                closest_corner->pt.y = (pt.y + closest_corner->pt.y) * 0.5f;

                cur_quad->count++;
                cur_quad->neighbors[i] = closest_quad;
                cur_quad->corners[i] = closest_corner;

                closest_quad->count++;
                closest_quad->neighbors[closest_corner_idx] = cur_quad;
				closest_quad->corners[closest_corner_idx] = closest_corner;
            }
        }
    }

// EVALUATE TIMER
#if TIMER
	float time = (float) (clock() - start_time) / CLOCKS_PER_SEC;
	FindQuadNeighbors2.open("timer/FindQuadNeighbors2.txt", ofstream::app);
	FindQuadNeighbors2 << "Time for mrFindQuadNeighbors2 was " << time << " seconds." << endl;
	FindQuadNeighbors2.close();
#endif
}



//===========================================================================
// AUGMENT PATTERN WITH ADDITIONAL QUADS
//===========================================================================
// The first part of the function is basically a copy of 
// "mrFindQuadNeighbors2"
// The comparisons between two points and two lines could be computed in their
// own function
static int mrAugmentBestRun( CvCBQuad *new_quads, int new_quad_count, int new_dilation, 
							  CvCBQuad **old_quads, int old_quad_count, int old_dilation )
{
//START TIMER
#if TIMER
	ofstream AugmentBestRun;
    time_t  start_time = clock();
#endif

	// thresh dilation is used to counter the effect of dilation on the
	// distance between 2 neighboring corners. Since the distance below is 
	// computed as its square, we do here the same. Additionally, we take the
	// conservative assumption that dilation was performed using the 3x3 CROSS
	// kernel, which coresponds to the 4-neighborhood.
	const float thresh_dilation = (float)(2*new_dilation+3)*(2*old_dilation+3)*2;	// the "*2" is for the x and y component
    int idx, i, k, j;																// the "3" is for initial corner mismatch
    float dx, dy, dist;
	

    // Search all old quads which have a neighbor that needs to be linked
    for( idx = 0; idx < old_quad_count; idx++ )
    {
        CvCBQuad* cur_quad = old_quads[idx];


        // For each corner of this quadrangle
        for( i = 0; i < 4; i++ )
        {
            CvPoint2D32f pt;
            float min_dist = FLT_MAX;
            int closest_corner_idx = -1;
            CvCBQuad *closest_quad = 0;
            CvCBCorner *closest_corner = 0;


			// If cur_quad corner[i] is already linked, continue
            if( cur_quad->corners[i]->needsNeighbor == false )
                continue;

            pt = cur_quad->corners[i]->pt;


            // Look for a match in all new_quads' corners
            for( k = 0; k < new_quad_count; k++ )
            {
				// Only look at unlabeled new quads
				if( new_quads[k].labeled == true)
					continue;

                for( j = 0; j < 4; j++ )
                {

					// Only proceed if they are less than dist away from each
					// other
                    dx = pt.x - new_quads[k].corners[j]->pt.x;
                    dy = pt.y - new_quads[k].corners[j]->pt.y;
                    dist = dx * dx + dy * dy;

                    if( (dist < min_dist) && 
						dist <= (cur_quad->edge_len + thresh_dilation) &&
                        dist <= (new_quads[k].edge_len + thresh_dilation) )
                    {
						// First Check everything from the viewpoint of the 
						// current quad compute midpoints of "parallel" quad 
						// sides 1
						float x1 = (cur_quad->corners[i]->pt.x + cur_quad->corners[(i+1)%4]->pt.x)/2;
						float y1 = (cur_quad->corners[i]->pt.y + cur_quad->corners[(i+1)%4]->pt.y)/2;				
						float x2 = (cur_quad->corners[(i+2)%4]->pt.x + cur_quad->corners[(i+3)%4]->pt.x)/2;
						float y2 = (cur_quad->corners[(i+2)%4]->pt.y + cur_quad->corners[(i+3)%4]->pt.y)/2;	
						// compute midpoints of "parallel" quad sides 2
						float x3 = (cur_quad->corners[i]->pt.x + cur_quad->corners[(i+3)%4]->pt.x)/2;
						float y3 = (cur_quad->corners[i]->pt.y + cur_quad->corners[(i+3)%4]->pt.y)/2;				
						float x4 = (cur_quad->corners[(i+1)%4]->pt.x + cur_quad->corners[(i+2)%4]->pt.x)/2;
						float y4 = (cur_quad->corners[(i+1)%4]->pt.y + cur_quad->corners[(i+2)%4]->pt.y)/2;	
						
						// MARTIN: Heuristic
						// For the corner "j" of quad "k" to be considered, 
						// it needs to be on the same side of the two lines as 
						// corner "i". This is given, if the cross product has 
						// the same sign for both computations below:
						float a1 = x1 - x2;
						float b1 = y1 - y2;
						// the current corner
						float c11 = cur_quad->corners[i]->pt.x - x2;
						float d11 = cur_quad->corners[i]->pt.y - y2;
						// the candidate corner
						float c12 = new_quads[k].corners[j]->pt.x - x2;
						float d12 = new_quads[k].corners[j]->pt.y - y2;
						float sign11 = a1*d11 - c11*b1;
						float sign12 = a1*d12 - c12*b1;

						float a2 = x3 - x4;
						float b2 = y3 - y4;
						// the current corner
						float c21 = cur_quad->corners[i]->pt.x - x4;
						float d21 = cur_quad->corners[i]->pt.y - y4;
						// the candidate corner
						float c22 = new_quads[k].corners[j]->pt.x - x4;
						float d22 = new_quads[k].corners[j]->pt.y - y4;
						float sign21 = a2*d21 - c21*b2;
						float sign22 = a2*d22 - c22*b2;

						// Also make sure that two border quads of the same row or
						// column don't link. Check from the current corner's view,
						// whether the corner diagonal from the candidate corner
						// is also on the same side of the two lines as the current
						// corner and the candidate corner.
						float c13 = new_quads[k].corners[(j+2)%4]->pt.x - x2;
						float d13 = new_quads[k].corners[(j+2)%4]->pt.y - y2;
						float c23 = new_quads[k].corners[(j+2)%4]->pt.x - x4;
						float d23 = new_quads[k].corners[(j+2)%4]->pt.y - y4;
						float sign13 = a1*d13 - c13*b1;
						float sign23 = a2*d23 - c23*b2;


						// Second: Then check everything from the viewpoint of
						// the candidate quad. Compute midpoints of "parallel"
						// quad sides 1
						float u1 = (new_quads[k].corners[j]->pt.x + new_quads[k].corners[(j+1)%4]->pt.x)/2;
						float v1 = (new_quads[k].corners[j]->pt.y + new_quads[k].corners[(j+1)%4]->pt.y)/2;				
						float u2 = (new_quads[k].corners[(j+2)%4]->pt.x + new_quads[k].corners[(j+3)%4]->pt.x)/2;
						float v2 = (new_quads[k].corners[(j+2)%4]->pt.y + new_quads[k].corners[(j+3)%4]->pt.y)/2;	
						// compute midpoints of "parallel" quad sides 2
						float u3 = (new_quads[k].corners[j]->pt.x + new_quads[k].corners[(j+3)%4]->pt.x)/2;
						float v3 = (new_quads[k].corners[j]->pt.y + new_quads[k].corners[(j+3)%4]->pt.y)/2;				
						float u4 = (new_quads[k].corners[(j+1)%4]->pt.x + new_quads[k].corners[(j+2)%4]->pt.x)/2;
						float v4 = (new_quads[k].corners[(j+1)%4]->pt.y + new_quads[k].corners[(j+2)%4]->pt.y)/2;	
						
						// MARTIN: Heuristic
						// For the corner "j" of quad "k" to be considered, 
						// it needs to be on the same side of the two lines as 
						// corner "i". This is given, if the cross product has 
						// the same sign for both computations below:
						float a3 = u1 - u2;
						float b3 = v1 - v2;
						// the current corner
						float c31 = cur_quad->corners[i]->pt.x - u2;
						float d31 = cur_quad->corners[i]->pt.y - v2;
						// the candidate corner
						float c32 = new_quads[k].corners[j]->pt.x - u2;
						float d32 = new_quads[k].corners[j]->pt.y - v2;
						float sign31 = a3*d31-c31*b3;
						float sign32 = a3*d32-c32*b3;

						float a4 = u3 - u4;
						float b4 = v3 - v4;
						// the current corner
						float c41 = cur_quad->corners[i]->pt.x - u4;
						float d41 = cur_quad->corners[i]->pt.y - v4;
						// the candidate corner
						float c42 = new_quads[k].corners[j]->pt.x - u4;
						float d42 = new_quads[k].corners[j]->pt.y - v4;
						float sign41 = a4*d41-c41*b4;
						float sign42 = a4*d42-c42*b4;

						// Also make sure that two border quads of the same row or
						// column don't link. Check from the candidate corner's view,
						// whether the corner diagonal from the current corner
						// is also on the same side of the two lines as the current
						// corner and the candidate corner.
						float c33 = cur_quad->corners[(i+2)%4]->pt.x - u2;
						float d33 = cur_quad->corners[(i+2)%4]->pt.y - v2;
						float c43 = cur_quad->corners[(i+2)%4]->pt.x - u4;
						float d43 = cur_quad->corners[(i+2)%4]->pt.y - v4;
						float sign33 = a3*d33-c33*b3;
						float sign43 = a4*d43-c43*b4;

						
						// This time we also need to make sure, that no quad
						// is linked to a quad of another dilation run which 
						// may lie INSIDE it!!!
						// Third: Therefore check everything from the viewpoint
						// of the current quad compute midpoints of "parallel" 
						// quad sides 1
						float x5 = cur_quad->corners[i]->pt.x;
						float y5 = cur_quad->corners[i]->pt.y;				
						float x6 = cur_quad->corners[(i+1)%4]->pt.x;
						float y6 = cur_quad->corners[(i+1)%4]->pt.y;	
						// compute midpoints of "parallel" quad sides 2
						float x7 = x5;
						float y7 = y5;				
						float x8 = cur_quad->corners[(i+3)%4]->pt.x;
						float y8 = cur_quad->corners[(i+3)%4]->pt.y;	
						
						// MARTIN: Heuristic
						// For the corner "j" of quad "k" to be considered, 
						// it needs to be on the other side of the two lines than 
						// corner "i". This is given, if the cross product has 
						// a different sign for both computations below:
						float a5 = x6 - x5;
						float b5 = y6 - y5;
						// the current corner
						float c51 = cur_quad->corners[(i+2)%4]->pt.x - x5;
						float d51 = cur_quad->corners[(i+2)%4]->pt.y - y5;
						// the candidate corner
						float c52 = new_quads[k].corners[j]->pt.x - x5;
						float d52 = new_quads[k].corners[j]->pt.y - y5;
						float sign51 = a5*d51 - c51*b5;
						float sign52 = a5*d52 - c52*b5;

						float a6 = x8 - x7;
						float b6 = y8 - y7;
						// the current corner
						float c61 = cur_quad->corners[(i+2)%4]->pt.x - x7;
						float d61 = cur_quad->corners[(i+2)%4]->pt.y - y7;
						// the candidate corner
						float c62 = new_quads[k].corners[j]->pt.x - x7;
						float d62 = new_quads[k].corners[j]->pt.y - y7;
						float sign61 = a6*d61 - c61*b6;
						float sign62 = a6*d62 - c62*b6;


						// Fourth: Then check everything from the viewpoint of 
						// the candidate quad compute midpoints of "parallel" 
						// quad sides 1
						float u5 = new_quads[k].corners[j]->pt.x;
						float v5 = new_quads[k].corners[j]->pt.y;				
						float u6 = new_quads[k].corners[(j+1)%4]->pt.x;
						float v6 = new_quads[k].corners[(j+1)%4]->pt.y;	
						// compute midpoints of "parallel" quad sides 2
						float u7 = u5;
						float v7 = v5;				
						float u8 = new_quads[k].corners[(j+3)%4]->pt.x;
						float v8 = new_quads[k].corners[(j+3)%4]->pt.y;	
						
						// MARTIN: Heuristic
						// For the corner "j" of quad "k" to be considered, 
						// it needs to be on the other side of the two lines than 
						// corner "i". This is given, if the cross product has 
						// a different sign for both computations below:
						float a7 = u6 - u5;
						float b7 = v6 - v5;
						// the current corner
						float c71 = cur_quad->corners[i]->pt.x - u5;
						float d71 = cur_quad->corners[i]->pt.y - v5;
						// the candidate corner
						float c72 = new_quads[k].corners[(j+2)%4]->pt.x - u5;
						float d72 = new_quads[k].corners[(j+2)%4]->pt.y - v5;
						float sign71 = a7*d71-c71*b7;
						float sign72 = a7*d72-c72*b7;

						float a8 = u8 - u7;
						float b8 = v8 - v7;
						// the current corner
						float c81 = cur_quad->corners[i]->pt.x - u7;
						float d81 = cur_quad->corners[i]->pt.y - v7;
						// the candidate corner
						float c82 = new_quads[k].corners[(j+2)%4]->pt.x - u7;
						float d82 = new_quads[k].corners[(j+2)%4]->pt.y - v7;
						float sign81 = a8*d81-c81*b8;
						float sign82 = a8*d82-c82*b8;





						// Check whether conditions are fulfilled
						if ( ((sign11 < 0 && sign12 < 0) || (sign11 > 0 && sign12 > 0))  && 
							 ((sign21 < 0 && sign22 < 0) || (sign21 > 0 && sign22 > 0))  &&
							 ((sign31 < 0 && sign32 < 0) || (sign31 > 0 && sign32 > 0))  &&   
							 ((sign41 < 0 && sign42 < 0) || (sign41 > 0 && sign42 > 0))	 &&	
							 ((sign11 < 0 && sign13 < 0) || (sign11 > 0 && sign13 > 0))  &&   
							 ((sign21 < 0 && sign23 < 0) || (sign21 > 0 && sign23 > 0))  &&
							 ((sign31 < 0 && sign33 < 0) || (sign31 > 0 && sign33 > 0))  &&   
							 ((sign41 < 0 && sign43 < 0) || (sign41 > 0 && sign43 > 0))  &&
							 ((sign51 < 0 && sign52 > 0) || (sign51 > 0 && sign52 < 0))  && 
							 ((sign61 < 0 && sign62 > 0) || (sign61 > 0 && sign62 < 0))  &&
							 ((sign71 < 0 && sign72 > 0) || (sign71 > 0 && sign72 < 0))  &&   
							 ((sign81 < 0 && sign82 > 0) || (sign81 > 0 && sign82 < 0)) )						
						{
							closest_corner_idx = j;
							closest_quad = &new_quads[k];
							min_dist = dist;
						}
                    }
                }
            }

            // Have we found a matching corner point?
            if( closest_corner_idx >= 0 && min_dist < FLT_MAX )
            {
                closest_corner = closest_quad->corners[closest_corner_idx];
                closest_corner->pt.x = (pt.x + closest_corner->pt.x) * 0.5f;
                closest_corner->pt.y = (pt.y + closest_corner->pt.y) * 0.5f;


                // We've found one more corner - remember it
				// ATTENTION: write the corner x and y coordinates separately, 
				// else the crucial row/column entries will be overwritten !!!
                cur_quad->corners[i]->pt.x = closest_corner->pt.x;
				cur_quad->corners[i]->pt.y = closest_corner->pt.y;
				cur_quad->neighbors[i] = closest_quad;
				closest_quad->corners[closest_corner_idx]->pt.x = closest_corner->pt.x;
				closest_quad->corners[closest_corner_idx]->pt.y = closest_corner->pt.y;
				
				
				// Label closest quad as labeled. In this way we exclude it
				// being considered again during the next loop iteration
				closest_quad->labeled = true;


				// We have a new member of the final pattern, copy it over
				old_quads[old_quad_count]				= new CvCBQuad;
				old_quads[old_quad_count]->count		= 1;
				old_quads[old_quad_count]->edge_len		= closest_quad->edge_len;
				old_quads[old_quad_count]->group_idx	= cur_quad->group_idx;	//the same as the current quad
				old_quads[old_quad_count]->labeled		= false;				//do it right afterwards
				
				
				// We only know one neighbor for sure, initialize rest with 
				// the NULL pointer
				old_quads[old_quad_count]->neighbors[closest_corner_idx]		= cur_quad;
				old_quads[old_quad_count]->neighbors[(closest_corner_idx+1)%4]	= NULL;
				old_quads[old_quad_count]->neighbors[(closest_corner_idx+2)%4]	= NULL;
				old_quads[old_quad_count]->neighbors[(closest_corner_idx+3)%4]	= NULL;

				for (int j = 0; j < 4; j++)
				{
					old_quads[old_quad_count]->corners[j]					= new CvCBCorner;
					old_quads[old_quad_count]->corners[j]->pt.x				= closest_quad->corners[j]->pt.x;
					old_quads[old_quad_count]->corners[j]->pt.y				= closest_quad->corners[j]->pt.y;
				}

				cur_quad->neighbors[i] = old_quads[old_quad_count];


				// Start the function again
				return -1;
            }
        }
    }
	
// EVALUATE TIMER
#if TIMER
	float time = (float) (clock() - start_time) / CLOCKS_PER_SEC;
	AugmentBestRun.open("timer/AugmentBestRun.txt", ofstream::app);
	AugmentBestRun << "Time for mrAugmentBestRun was " << time << " seconds." << endl;
	AugmentBestRun.close();
#endif

	// Finished, don't start the function again
	return 1;
}



//===========================================================================
// GENERATE QUADRANGLES
//===========================================================================
static int
icvGenerateQuads( CvCBQuad **out_quads, CvCBCorner **out_corners,
                  CvMemStorage *storage, CvMat *image, int flags, int dilation, bool firstRun )
{
//START TIMER
#if TIMER
	ofstream GenerateQuads;
    time_t  start_time = clock();
#endif

	// Initializations
    int quad_count = 0;
    CvMemStorage *temp_storage = 0;

    if( out_quads )
        *out_quads = 0;

    if( out_corners )
        *out_corners = 0;

    CV_FUNCNAME( "icvGenerateQuads" );

    __CV_BEGIN__;

    CvSeq *src_contour = 0;
    CvSeq *root;
    CvContourEx* board = 0;
    CvContourScanner scanner;
    int i, idx, min_size;

    CV_ASSERT( out_corners && out_quads );


    // Empiric sower bound for the size of allowable quadrangles.
	// MARTIN, modified: Added "*0.1" in order to find smaller quads.
	min_size = cvRound( image->cols * image->rows * .03 * 0.01 * 0.92 * 0.1);


    // Create temporary storage for contours and the sequence of pointers to
	// found quadrangles
    CV_CALL( temp_storage = cvCreateChildMemStorage( storage ));
    CV_CALL( root = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSeq*), temp_storage ));


    // Initialize contour retrieving routine
    CV_CALL( scanner = cvStartFindContours( image, temp_storage, sizeof(CvContourEx),
                                            CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ));


    // Get all the contours one by one
    while( (src_contour = cvFindNextContour( scanner )) != 0 )
    {
        CvSeq *dst_contour = 0;
        CvRect rect = ((CvContour*)src_contour)->rect;
		
	

        // Reject contours with a too small perimeter and contours which are 
		// completely surrounded by another contour
		// MARTIN: If this function is called during PART 1, then the parameter "first run"
		// is set to "true". This guarantees, that only "nice" squares are detected.
		// During PART 2, we allow the polygonial approcimation function below to
		// approximate more freely, which can result in recognized "squares" that are in
		// reality multiple blurred and sticked together squares.
        if( CV_IS_SEQ_HOLE(src_contour) && rect.width*rect.height >= min_size )
        {
            int min_approx_level = 2, max_approx_level;
			if (firstRun == true)
				max_approx_level = 3;
			else
				max_approx_level = MAX_CONTOUR_APPROX;
            int approx_level;
            for( approx_level = min_approx_level; approx_level <= max_approx_level; approx_level++ )
            {
                dst_contour = cvApproxPoly( src_contour, sizeof(CvContour), temp_storage,
                                            CV_POLY_APPROX_DP, (float)approx_level );
                
				
				// We call this again on its own output, because sometimes
                // cvApproxPoly() does not simplify as much as it should.
                dst_contour = cvApproxPoly( dst_contour, sizeof(CvContour), temp_storage,
                                            CV_POLY_APPROX_DP, (float)approx_level );

				if( dst_contour->total == 4 )
                    break;
            }


            // Reject non-quadrangles
            if(dst_contour->total == 4 && cvCheckContourConvexity(dst_contour) )
            {
              CvPoint pt[4];
              double d1, d2, p = cvContourPerimeter(dst_contour);
              double area = fabs(cvContourArea(dst_contour, CV_WHOLE_SEQ));
              double dx, dy;

              for( i = 0; i < 4; i++ )
                pt[i] = *(CvPoint*)cvGetSeqElem(dst_contour, i);

              dx = pt[0].x - pt[2].x;
              dy = pt[0].y - pt[2].y;
              d1 = sqrt(dx*dx + dy*dy);

              dx = pt[1].x - pt[3].x;
              dy = pt[1].y - pt[3].y;
              d2 = sqrt(dx*dx + dy*dy);

              // PHILIPG: Only accept those quadrangles which are more
              // square than rectangular and which are big enough
              double d3, d4;
              dx = pt[0].x - pt[1].x;
              dy = pt[0].y - pt[1].y;
              d3 = sqrt(dx*dx + dy*dy);
              dx = pt[1].x - pt[2].x;
              dy = pt[1].y - pt[2].y;
              d4 = sqrt(dx*dx + dy*dy);
              if(true)//!(flags & CV_CALIB_CB_FILTER_QUADS) ||
                //d3*4 > d4 && d4*4 > d3 && d3*d4 < area*1.5 && area > min_size &&
                //d1 >= 0.15 * p && d2 >= 0.15 * p )
              {
                CvContourEx* parent = (CvContourEx*)(src_contour->v_prev);
                parent->counter++;
                if( !board || board->counter < parent->counter )
                  board = parent;
                dst_contour->v_prev = (CvSeq*)parent;
                cvSeqPush( root, &dst_contour );
              }
            }
        }
    }


    // Finish contour retrieving
    cvEndFindContours( &scanner );


    // Allocate quad & corner buffers
    CV_CALL( *out_quads = (CvCBQuad*)cvAlloc(root->total * sizeof((*out_quads)[0])));
    CV_CALL( *out_corners = (CvCBCorner*)cvAlloc(root->total * 4 * sizeof((*out_corners)[0])));


    // Create array of quads structures
    for( idx = 0; idx < root->total; idx++ )
    {
        CvCBQuad* q = &(*out_quads)[quad_count];
        src_contour = *(CvSeq**)cvGetSeqElem( root, idx );
        if( (flags & CV_CALIB_CB_FILTER_QUADS) && src_contour->v_prev != (CvSeq*)board )
            continue;


        // Reset group ID
        memset( q, 0, sizeof(*q) );
        q->group_idx = -1;
        assert( src_contour->total == 4 );
        for( i = 0; i < 4; i++ )
        {
            CvPoint2D32f pt = cvPointTo32f(*(CvPoint*)cvGetSeqElem(src_contour, i));
            CvCBCorner* corner = &(*out_corners)[quad_count*4 + i];

            memset( corner, 0, sizeof(*corner) );
            corner->pt = pt;
            q->corners[i] = corner;
        }
        q->edge_len = FLT_MAX;
        for( i = 0; i < 4; i++ )
        {
            float dx = q->corners[i]->pt.x - q->corners[(i+1)&3]->pt.x;
            float dy = q->corners[i]->pt.y - q->corners[(i+1)&3]->pt.y;
            float d = dx*dx + dy*dy;
            if( q->edge_len > d )
                q->edge_len = d;
        }
        quad_count++;
    }

    __CV_END__;

    if( cvGetErrStatus() < 0 )
    {
        if( out_quads )
            cvFree( out_quads );
        if( out_corners )
            cvFree( out_corners );
        quad_count = 0;
    }

    cvReleaseMemStorage( &temp_storage );

// EVALUATE TIMER
#if TIMER
	float time = (float) (clock() - start_time) / CLOCKS_PER_SEC;
	GenerateQuads.open("timer/GenerateQuads.txt", ofstream::app);
	GenerateQuads << "Time for icvGenerateQuads was " << time << " seconds." << endl;
	GenerateQuads.close();
#endif

    return quad_count;
}



//===========================================================================
// WRITE CORNERS TO FILE
//===========================================================================
static int mrWriteCorners( CvCBQuad **output_quads, int count, CvSize pattern_size, int min_number_of_corners )
{
	// Initialize
	int corner_count = 0;
	bool flagRow = false;
	bool flagColumn = false;
	int maxPattern_sizeRow = -1;
	int maxPattern_sizeColumn = -1;


	// Return variable
	int internal_found = 0;
	

	// Compute the minimum and maximum row / column ID
	// (it is unlikely that more than 8bit checkers are used per dimension)
	int min_row		=  127;
	int max_row		= -127;
	int min_column	=  127;
	int max_column	= -127;

	for(int i = 0; i < count; i++ )
    {
		CvCBQuad* q = output_quads[i];
		
		for(int j = 0; j < 4; j++ )
		{
			if( (q->corners[j])->row > max_row)
				max_row = (q->corners[j])->row;
			if( (q->corners[j])->row < min_row)
				min_row = (q->corners[j])->row;
			if( (q->corners[j])->column > max_column)
				max_column = (q->corners[j])->column;
			if( (q->corners[j])->column < min_column)
				min_column = (q->corners[j])->column;
		}
	}


	// If in a given direction the target pattern size is reached, we know exactly how
	// the checkerboard is oriented.
	// Else we need to prepare enought "dummy" corners for the worst case.
	for(int i = 0; i < count; i++ )
    {
		CvCBQuad* q = output_quads[i];
		
		for(int j = 0; j < 4; j++ )
		{
			if( (q->corners[j])->column == max_column && (q->corners[j])->row != min_row && (q->corners[j])->row != max_row )
			{
				if( (q->corners[j]->needsNeighbor) == false)
				{
					// We know, that the target pattern size is reached
					// in column direction
					flagColumn = true;
				}
			}
			if( (q->corners[j])->row == max_row && (q->corners[j])->column != min_column && (q->corners[j])->column != max_column )
			{
				if( (q->corners[j]->needsNeighbor) == false)
				{
					// We know, that the target pattern size is reached
					// in row direction
					flagRow = true;
				}
			}
		}
	}

	if( flagColumn == true)
	{
		if( max_column - min_column == pattern_size.width + 1)
		{
			maxPattern_sizeColumn = pattern_size.width;
			maxPattern_sizeRow = pattern_size.height;
		}
		else
		{
			maxPattern_sizeColumn = pattern_size.height;
			maxPattern_sizeRow = pattern_size.width;
		}
	}
	else if( flagRow == true)
	{
		if( max_row - min_row == pattern_size.width + 1)
		{
			maxPattern_sizeRow = pattern_size.width;
			maxPattern_sizeColumn = pattern_size.height;
		}
		else
		{
			maxPattern_sizeRow = pattern_size.height;
			maxPattern_sizeColumn = pattern_size.width;
		}
	}
	else
	{
		// If target pattern size is not reached in at least one of the two
		// directions,  then we do not know where the remaining corners are
		// located. Account for this.
		maxPattern_sizeColumn = max(pattern_size.width, pattern_size.height);
		maxPattern_sizeRow = max(pattern_size.width, pattern_size.height);
	}


	// Open the output files
	ofstream cornersX("cToMatlab/cornersX.txt");
	ofstream cornersY("cToMatlab/cornersY.txt");
	ofstream cornerInfo("cToMatlab/cornerInfo.txt");


	// Write the corners in increasing order to the output file
	for(int i = min_row + 1; i < maxPattern_sizeRow + min_row + 1; i++)
	{
		for(int j = min_column + 1; j < maxPattern_sizeColumn + min_column + 1; j++)
		{
			// Reset the iterator
			int iter = 1;

			for(int k = 0; k < count; k++)
			{
				for(int l = 0; l < 4; l++)
				{
					if(((output_quads[k])->corners[l]->row == i) && ((output_quads[k])->corners[l]->column == j) )
					{
						// Only write corners to the output file, which are connected
						// i.e. only if iter == 2
						if( iter == 2)
						{
							// The respective row and column have been found, print it to
							// the output file, only do this once
							cornersX << (output_quads[k])->corners[l]->pt.x;
							cornersX << " ";
							cornersY << (output_quads[k])->corners[l]->pt.y;
							cornersY << " ";

							corner_count++;
						}
						

						// If the iterator is larger than two, this means that more than
						// two corners have the same row / column entries. Then some
						// linking errors must have occured and we should not use the found
						// pattern
						if (iter > 2)
							return -1;

						iter++;
					}
				}
			}

			// If the respective row / column is non - existent or is a border corner
			if (iter == 1 || iter == 2)
			{
				cornersX << -1;
				cornersX << " ";
				cornersY << -1;
				cornersY << " ";
			}
		}
		cornersX << endl;
		cornersY << endl;
	}


	// Write to the corner matrix size info file
	cornerInfo << maxPattern_sizeRow<< " " << maxPattern_sizeColumn << endl;


	// Close the output files
	cornersX.close(); 
	cornersY.close();
	cornerInfo.close();


	// check whether enough corners have been found
	if (corner_count >= min_number_of_corners)
		internal_found = 1;
	else
		internal_found = 0;


	// pattern found, or not found?
	return internal_found;
}



//===========================================================================
// getUVPoints
//===========================================================================
static void getUVPoints(const CvMat *i_imCurr, std::vector<cv::Point2i> &outvec, int threshVal) {
  cv::Mat *m_imCurr = new cv::Mat(cv::Size(i_imCurr->cols, i_imCurr->rows), i_imCurr->type, i_imCurr->data.ptr);

  std::vector< cv::Point > fastPoints;
  std::vector< cv::Point > fastInterior;
  initFAST(fastPoints, fastInterior);

  if (DEBUG)
    std::cout << "Thresh: " << threshVal << std::endl;
  cv::Rect m_roi     = cv::Rect(cv::Point(0, 0), m_imCurr->size());
  cv::Mat m_imCheck = cv::Mat(m_imCurr->size(), CV_8UC1);
  m_imCheck = cv::Scalar(0);
  cv::Point peakPoint;
  int x,y;

  bool          test;
  unsigned char maximumVal = 0;
  for (int j = 0; j < m_imCurr->rows; j++) {
    for (int i = 0; i < m_imCurr->cols; i++) {
          /* std::cout << "imCheck at [" << i << ":" << j << "]: " << m_imCheck.data[index2d(i, j)] << std::endl; */
      if (m_imCheck.data[index2d(i, j)] == 0) {
        /* if (m_imCurr->data[index2d(i, j)] > threshVal) { */
        if (true) {
          test   = true;
          /* std::cout << "Fast points size:" << fastPoints.size() << std::endl; */
          for (int m = 0; m < (int)(fastPoints.size()); m++) {
            x = i + fastPoints[m].x;
            if (x < 0) {
              test = false;
              break;
            }
            if (x >= m_roi.width) {
              test = false;
              break;
            }

            y = j + fastPoints[m].y;
            if (y < 0) {
              test = false;
              break;
            }
            if (y >= m_roi.height) {
              test = false;
              break;
            }

            if (m_imCheck.data[index2d(x,y)] == 255){
              test =false;
              break;
            }


            if ((m_imCurr->data[index2d(i, j)] - m_imCurr->data[index2d(x, y)]) < (threshVal/2)) {

              /* if (DEBUG) */
              /*   std::cout << "Fast test failed for: [" << x << ":" << y << "] with diff. of" << (m_imCurr->data[index2d(i, j)] - m_imCurr->data[index2d(x, y)])<< std::endl; */
              test = false;
              break;
            }
          }
          /* if (DEBUG) */
          /*   std::cout << "Fast test passed for: [" << x << ":" << y << "]" << std::endl; */

          if (test) {
            maximumVal = 0;
            for (int m = 0; m < (int)(fastInterior.size()); m++) {
            /* for (int m = 0; m < 1; m++) { */
              x = i + fastInterior[m].x;
              if (x < 0) {
                continue;
              }
              if (x >= m_roi.width) {
                continue;
              }

              y = j + fastInterior[m].y;
              if (y < 0) {
                continue;
              }
              if (y >= m_roi.height) {
                continue;
              }
              /* std::cout << "here: " << x << ":" << y << std::endl; */

              if (m_imCheck.data[index2d(x, y)] == 0) {
                if (m_imCurr->data[index2d(x, y)] > maximumVal) {
                  maximumVal  = m_imCurr->data[index2d(x, y)];
                  peakPoint.x = x;
                  peakPoint.y = y;
                }
                m_imCheck.data[index2d(x, y)] = 255;
              }
            }
            /* std::cout << "peakPoint: "<< peakPoint << std::endl; */
            outvec.push_back(peakPoint);
          }
        }
      }
    }
  }
  return;
}

static void initFAST(std::vector<cv::Point> &fastPoints, std::vector<cv::Point> &fastInterior) {
  fastPoints.clear();

  fastPoints.push_back(cv::Point(0, -3));
  fastPoints.push_back(cv::Point(0, 3));
  fastPoints.push_back(cv::Point(3, 0));
  fastPoints.push_back(cv::Point(-3, 0));

  fastPoints.push_back(cv::Point(2, -2));
  fastPoints.push_back(cv::Point(-2, 2));
  fastPoints.push_back(cv::Point(-2, -2));
  fastPoints.push_back(cv::Point(2, 2));

  fastPoints.push_back(cv::Point(-1, -3));
  fastPoints.push_back(cv::Point(1, 3));
  fastPoints.push_back(cv::Point(3, -1));
  fastPoints.push_back(cv::Point(-3, 1));

  fastPoints.push_back(cv::Point(1, -3));
  fastPoints.push_back(cv::Point(-1, 3));
  fastPoints.push_back(cv::Point(3, 1));
  fastPoints.push_back(cv::Point(-3, -1));

  fastInterior.clear();

  /* fastInterior.push_back(cv::Point(-1, -2)); */
  /* fastInterior.push_back(cv::Point(0, -2)); */
  /* fastInterior.push_back(cv::Point(1, -2)); */

  /* fastInterior.push_back(cv::Point(-2, -1)); */
  /* fastInterior.push_back(cv::Point(-1, -1)); */
  /* fastInterior.push_back(cv::Point(0, -1)); */
  /* fastInterior.push_back(cv::Point(1, -1)); */
  /* fastInterior.push_back(cv::Point(2, -1)); */

  /* fastInterior.push_back(cv::Point(-2, 0)); */
  /* fastInterior.push_back(cv::Point(-1, 0)); */
  fastInterior.push_back(cv::Point(0, 0));
  fastInterior.push_back(cv::Point(1, 0));
  fastInterior.push_back(cv::Point(2, 0));

  /* fastInterior.push_back(cv::Point(-2, 1)); */
  /* fastInterior.push_back(cv::Point(-1, 1)); */
  fastInterior.push_back(cv::Point(0, 1));
  fastInterior.push_back(cv::Point(1, 1));
  fastInterior.push_back(cv::Point(2, 1));

  /* fastInterior.push_back(cv::Point(-1, 2)); */
  fastInterior.push_back(cv::Point(0, 2));
  fastInterior.push_back(cv::Point(1, 2));
}


double distance(cv::Point a, cv::Point b){
return cv::norm(a-b);
}
double distance(cv::Point* a, cv::Point* b){
return distance(*a,*b);
}
double distance(cv::Point a, cv::Point* b){
return distance(a,*b);
}

double cosAngle(cv::Point *a,cv::Point *b,cv::Point *c){
  cv::Point da = (*b)-(*a);
  cv::Point db = (*c)-(*b);
  /* std::cout << "da: " << da << std::endl; */
  /* std::cout << "db: " << db << std::endl; */
  double output = (da.dot(db))/(cv::norm(da)*cv::norm(db));
  /* std::cout << "cosAngle: " << output << std::endl; */
  return output;
}
double angle3p(cv::Point *a,cv::Point *b,cv::Point *c){
  cv::Point da = -(*b)+(*a);
  cv::Point db = (*c)-(*b);
    const double sin_ab = da.cross(db);
    const double cos_ab = da.dot(db);
    double angle = -std::atan2(sin_ab, cos_ab);
  /* std::cout << "da: " << da << std::endl; */
  /* std::cout << "db: " << db << std::endl; */
  if (angle<0)
    angle+=2*M_PI;
  /* double output = (da.dot(db))/(cv::norm(da)*cv::norm(db)); */
  /* std::cout << "angle: " << angle << std::endl; */
  return angle;
}

#define maxDistSample 7.0
double getMaxDistance(cv::Point* currPoint, std::vector<cv::Point> & uv_points){
  std::vector<int> ordered;
  for (int i=0;i<(int)(uv_points.size());i++){
    ordered.push_back(i);
  }

  for (size_t i=0; i< uv_points.size(); i++){
    for (size_t j=0; j< uv_points.size()-1; j++){
      if (distance(uv_points[ordered[j]], currPoint)>distance(uv_points[ordered[j+1]], currPoint)){
        std::swap(ordered[j],ordered[j+1]);
      }
    }
  }
  /* std::cout << "Order of closest to " << *currPoint << ": " << std::endl; */
  for (int &i : ordered){
    /* std::cout << uv_points[i] << std::endl; */
  }

  double sumDist = 0;
  for (int i=1; i<maxDistSample+1; i++){
    sumDist += distance(uv_points[ordered[i]], currPoint);
  }
  double distAvg = 1.5*sumDist/maxDistSample;

  /* double bestDist = distance(uv_points[ordered[maxDistSample]], currPoint); */
  /* std::cout << "Best distance is " << bestDist << ": " << uv_points[ordered[3]] << std::endl; */
  /* std::cout << "Average distance is " << distAvg << std::endl; */

  return distAvg;
}

int findLeftmostCorner(std::vector<cv::Point2i> &uv_points){
  int leftMost = 0;
  for (int i=0; i< (int)(uv_points.size()); i++){
    if (uv_points[i].x < uv_points[leftMost].x)
      leftMost = i;
  }
  return leftMost;
}


static void getHullPoints(std::vector<cv::Point2i> &uv_points,  HullPoint*& start, double maxConcaveAngle, double similarAngle, CvMat* vis_img){
  std::vector<bool> marked_points;
  for (auto p : uv_points)
    marked_points.push_back(false);
  int leftMostCornerIndex = findLeftmostCorner(uv_points);
  start = new HullPoint(&(uv_points[leftMostCornerIndex]));
  HullPoint* currHullPoint = start;
  marked_points[leftMostCornerIndex] = true;
  //VISUALIZATION--------------------------------------------------------------
#if VIS
  cvCircle(vis_img, uv_points[leftMostCornerIndex], 7, cv::Scalar(255));
  cvCircle(vis_img, uv_points[leftMostCornerIndex], 3, cv::Scalar(255));
  cvNamedWindow( "ocv_Markers", 1 );
  cvShowImage( "ocv_Markers", vis_img);
  cvWaitKey(30);
#endif
//END------------------------------------------------------------------------
  cv::Point initPoint = uv_points[leftMostCornerIndex];
  initPoint.y = initPoint.y+10;
  cv::Point* prevPoint = &(initPoint);
  cv::Point* tentPoint;
  cv::Point* currPoint = currHullPoint->point;
  bool done =false;
  do {
    /* std::cout << "Addresses: " << currHullPoint->point << " : " << start->point << std::endl; */
    double maxDistance = getMaxDistance(currPoint, uv_points);
    double bestAngle = 0.0;
    double distOfBest = 9999.0;
    double currAngle;
    int bestIndex = -1;
    for (int i=0; i<(int)(marked_points.size()); i++){
      if (&(uv_points[i]) == currPoint)
        continue;

      if ((marked_points[i] == true) && (&(uv_points[i]) != start->point))
        continue;

      tentPoint = &(uv_points[i]);
      double distCurr = distance(currPoint,tentPoint);
      if (distCurr > maxDistance){
        if (DEBUG)
          std::cout << "Distance check for point " << *tentPoint << " failed with dist. of " << distance(currPoint,tentPoint) << " vs " << maxDistance << std::endl;
        continue;
      }
      if (DEBUG)
        std::cout << "Distance check for point " << *tentPoint << " passed with dist. of " << distance(currPoint,tentPoint) << " vs " << maxDistance << std::endl;

      currAngle = angle3p(prevPoint, currPoint, tentPoint );
      if (!(currAngle>(M_PI+maxConcaveAngle))){
        if ( 
            ((fabs(currAngle-bestAngle)<(similarAngle))&&  (distCurr < distOfBest) ) || 
            (currAngle>(bestAngle+similarAngle))
           ){
          if (DEBUG)
            std::cout << "Accepting point " << *tentPoint << " as the current best at angle " << currAngle << " and distance " << distCurr << std::endl;
          bestAngle = currAngle;
          distOfBest = distCurr;
          bestIndex = i;
        }
      }
      else
        if (DEBUG)
          std::cout << "Skipped point " << *tentPoint << " for representing large concavity at angle of " << currAngle << std::endl;
    }
    if (DEBUG)
      std::cout << "Selected point " << uv_points[bestIndex] << std::endl;
    //VISUALIZATION--------------------------------------------------------------
#if VIS
    cvCircle(vis_img, uv_points[bestIndex], 7, cv::Scalar(255));
    cvNamedWindow( "ocv_Markers", 1 );
    cvShowImage( "ocv_Markers", vis_img);
		cvWaitKey(DEBUG?0:10);
#endif
//END------------------------------------------------------------------------
//

    if (&(uv_points[bestIndex]) == start->point)
      done = true;

    if (!done)
      currHullPoint->right = new HullPoint(&(uv_points[bestIndex]));
    else 
      currHullPoint->right = start;

    currHullPoint->right->left = currHullPoint;
    currHullPoint->angle = bestAngle;
    marked_points.at(bestIndex) = true;
    if (done)
      break;
    prevPoint = currHullPoint->point;
    currHullPoint = currHullPoint->right;
    currPoint = currHullPoint->point;
  }while (true);
  /* }while (currHullPoint->point != start->point); */
  return;

}

static HullPoint* getSharpestHullPoint(HullPoint* start){
  double angle = 2*M_PI;
  HullPoint *currPoint = start;
  HullPoint *sharpestPoint = 0;
  do {
    if (currPoint->angle < angle){
      angle = currPoint->angle;
      sharpestPoint = currPoint;
    }
    currPoint = currPoint->right;
  } while (currPoint != start);
  return sharpestPoint;
}

cv::Point2d multiply(cv::Mat M, cv::Point2i p){
  cv::Point2d output;
  output.x = M.at<float>(cv::Point(0,0))*p.x + M.at<float>(cv::Point(1,0))*p.y;
  output.y = M.at<float>(cv::Point(0,1))*p.x + M.at<float>(cv::Point(1,1))*p.y;
  return output;
}

static void getGrid(HullPoint* corner_hull_point, GridPoint*& grid_points, bool& x_axis_first, std::vector<cv::Point2i> &uv_points,  int W, int H, CvMat *vis_img){
  std::vector<bool> marked_points;
  for (auto p : uv_points)
    marked_points.push_back(false);
  if (DEBUG)
    std::cout << "corner point is :" << *(corner_hull_point->point) << std::endl;
  HullPoint* hull_point_current = corner_hull_point;
  grid_points = new GridPoint(hull_point_current->point);
  GridPoint* grid_point_current = grid_points;
  GridPoint* grid_point_first_in_axis = grid_points;
  HullPoint* hull_point_first_in_axis = corner_hull_point;
  if (DEBUG)
    std::cout << "current grid point " << *(grid_point_current->point) << std::endl;
  grid_point_current->right = new GridPoint(hull_point_current->right->point);
  grid_point_current->below = new GridPoint(hull_point_current->left->point);
  grid_point_current->right->left = grid_point_current;
  grid_point_current->below->above = grid_point_current;

  bool first_line = true;
  HullPoint* hull_point_potential_end_a = hull_point_current;
  HullPoint* hull_point_potential_end_b = hull_point_current;
  HullPoint* hull_point_last = hull_point_current;
  for (int i=0; i<W-1; i++){
    hull_point_potential_end_a = hull_point_potential_end_a->right;
  }
  for (int i=0; i<H-1; i++){
    hull_point_potential_end_b = hull_point_potential_end_b->right;
  }
  for (int i=0; i<(H+W-2); i++){
    hull_point_last = hull_point_last->right;
  }

  /* std::cout << "a: " <<  *hull_point_potential_end_a->point << std::endl; */
  /* std::cout << "b: " <<  *hull_point_potential_end_b->point << std::endl; */
  /* cv::waitKey(0); */

  cv::Mat transformer(cv::Size(2,2), CV_32F);
  do {
    do {
      if (DEBUG){
        std::cout << "start point: " <<  *(grid_point_current->point) << std::endl;
        std::cout << "right point: " <<  *(grid_point_current->right->point) << std::endl;
        std::cout << "below point: " <<  *(grid_point_current->below->point) << std::endl;
      }

      auto xvec = *(grid_point_current->right->point) - *(grid_point_current->point);
      auto yvec = *(grid_point_current->below->point) - *(grid_point_current->point);
      transformer.at<float>(cv::Point(0,0)) = xvec.x;
      transformer.at<float>(cv::Point(1,0)) = yvec.x;
      transformer.at<float>(cv::Point(0,1)) = xvec.y;
      transformer.at<float>(cv::Point(1,1)) = yvec.y;
      if (DEBUG)
        std::cout << "inverted transformer" << std::endl << transformer <<std::endl;
      transformer = transformer.inv();
      if (DEBUG)
        std::cout << "using transformer" << std::endl << transformer <<std::endl;
      cv::Point2d trans_point;
      double distance_from_ideal_last, distance_from_ideal_current;
      bool found_one = false;
      for (int i=0; i<(int)(uv_points.size()); i++){
        if ((marked_points[i])){
          /* bool at_end = false; */
          /* /1* if (hull_point_potential_end_a != nullptr) *1/ */
          /* /1*   if (&(uv_points[i])==hull_point_potential_end_a->right->point) *1/ */
          /* /1*     at_end = true; *1/ */
          /* /1* if (hull_point_potential_end_b != nullptr) *1/ */
          /* /1*   if (&(uv_points[i])==hull_point_potential_end_b->right->point) *1/ */
          /* /1*     at_end = true; *1/ */

          /* /1* if (at_end) *1/ */
          /* /1*   std::cout << "at end" << std::endl; *1/ */

          /* if (!at_end){ */
            if (DEBUG)
              std::cout << "uv_point " << uv_points[i] << " is marked, skipping it." << std::endl;
          /* } */
          continue;
        }
        trans_point = multiply(transformer,(uv_points[i]-*(grid_point_current->point)));
        if (DEBUG)
          std::cout << "uv_point " << uv_points[i] << " shifted to " << (uv_points[i]-*(grid_point_current->point)) << " transformed to "<< trans_point << std::endl;
        if ((trans_point.x > 0.5) && (trans_point.x < 1.5))
          if ((trans_point.y > 0.5) && (trans_point.y < 1.5)){
            if (DEBUG)
              std::cout << "Found new grid point at:" << uv_points[i] << std::endl;


            distance_from_ideal_current = cv::norm(trans_point-cv::Point2d(1.0,1.0));
            if ((!found_one) || (found_one && (distance_from_ideal_current<distance_from_ideal_last)) )
            {
              //VISUALIZATION--------------------------------------------------------------
#if VIS
              cvCircle(vis_img, uv_points[i], 12, cv::Scalar(255));
              cvNamedWindow( "ocv_Markers", 1 );
              cvShowImage( "ocv_Markers", vis_img);
              cvWaitKey(DEBUG?0:10);
#endif
              //END------------------------------------------------------------------------
              grid_point_current->right->below = new GridPoint(&(uv_points[i]));
              grid_point_current->below->right = grid_point_current->right->below;
              grid_point_current->right->below->above = grid_point_current->right;
              grid_point_current->right->below->left = grid_point_current->below;
              marked_points[i] = true;
              distance_from_ideal_last = distance_from_ideal_current;
              found_one = true;
            }

            if (&(uv_points[i]) == hull_point_last->point){
              if (DEBUG)
                std::cout << "Finished grid extraction!" << std::endl;
              return;
            }
          }
      }
      if (!found_one){
        if (hull_point_potential_end_a != nullptr)
          if (grid_point_current->point == hull_point_potential_end_a->point){
            x_axis_first = true;
            hull_point_potential_end_a = hull_point_potential_end_a->right;
            hull_point_potential_end_b = nullptr;
            delete grid_point_current->right;
            grid_point_current->right = nullptr;
            /* std::cout << "a: " <<  *hull_point_potential_end_a->point << std::endl; */
            /* cv::waitKey(0); */
            if (DEBUG)
              std::cout << "Reached the end of line by X" << std::endl;
            break;
          }
        if (hull_point_potential_end_b != nullptr)
          if (grid_point_current->point == hull_point_potential_end_b->point){
            x_axis_first = false;
            hull_point_potential_end_b = hull_point_potential_end_b->right;
            hull_point_potential_end_a = nullptr;
            delete grid_point_current->right;
            grid_point_current->right = nullptr;
            if (DEBUG)
              std::cout << "Reached the end of line by Y" << std::endl;
            /* std::cout << "b: " <<  *hull_point_potential_end_b->point << std::endl; */
            /* cv::waitKey(0); */
            break; }

        /* std::cout << "Could not find a valid next point!" << std::endl; */
        /* std::cout << "b: " <<  *hull_point_potential_end_b->point << std::endl; */
        /* std::raise(SIGSEGV); */
        /* cv::waitKey(0); */
      }

      hull_point_current = hull_point_current->right;
      grid_point_current = grid_point_current->right;


      if (first_line){
        grid_point_current->right = new GridPoint(hull_point_current->right->point);
        grid_point_current->right->left = grid_point_current;
      }

      if (grid_point_current->right == nullptr){
        if (x_axis_first)
          hull_point_potential_end_a = hull_point_potential_end_a->right;
        else
          hull_point_potential_end_b = hull_point_potential_end_b->right;
        break;
      }


    } while (true);
    first_line = false;
    if (hull_point_current == hull_point_last){
      break;
    }

    if (DEBUG)
      std::cout << "Next line" <<std::endl;

    hull_point_current = hull_point_first_in_axis->left;
    grid_point_current = grid_point_first_in_axis->below;
    /* grid_point_current->right = new GridPoint(grid_point_first_in_axis->right->below->point); */
    grid_point_current->below = new GridPoint(hull_point_current->left->point);
    /* grid_point_current->right->left = grid_point_current; */
    hull_point_first_in_axis = hull_point_current;
    grid_point_first_in_axis = grid_point_current;


  } while (true);

  if (DEBUG)
    std::cout << "Invalid state!" <<std::endl;

}

static int mrWriteMarkers(GridPoint*& grid_points, bool x_axis_first,  CvSize pattern_size, int min_number_of_corners ){
  GridPoint* grid_point_current;
  GridPoint* grid_point_line_first;

  // Open the output files
  ofstream cornersX("cToMatlab/cornersX.txt");
	ofstream cornersY("cToMatlab/cornersY.txt");
	ofstream cornerInfo("cToMatlab/cornerInfo.txt");

  grid_point_current = grid_points;
  grid_point_line_first = grid_point_current;

  for (int j=0; j<pattern_size.height; j++){
    for (int i=0; i<pattern_size.width; i++){
      cornersX << grid_point_current->point->x;
      cornersX << " ";
      cornersY << grid_point_current->point->y;
      cornersY << " ";
      if (x_axis_first)
        grid_point_current = grid_point_current->right;
      else
        grid_point_current = grid_point_current->below;
    }
		cornersX << endl;
		cornersY << endl;
    if (x_axis_first){
      grid_point_current = grid_point_line_first->below;
      grid_point_line_first = grid_point_line_first->below;
    }
    else{
      grid_point_current = grid_point_line_first->right;
      grid_point_line_first = grid_point_line_first->right;
    }
  }



	// Write to the corner matrix size info file
	cornerInfo << pattern_size.width << " " << pattern_size.height << endl;


	// Close the output files
	cornersX.close(); 
	cornersY.close();
	cornerInfo.close();


	// pattern found, or not found?
	return 1;
}
//===========================================================================
// END OF FILE
//===========================================================================
