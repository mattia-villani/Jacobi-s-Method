#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<iostream>
#include<pthread.h>
#include<vector>
#include<stack>
#include<ctime>
#include<ff/pipeline.hpp>
#include<ff/farm.hpp>
#include<ff/parallel_for.hpp>
using namespace std;
using namespace ff;

// _t used in the place of float to easly change it to double
typedef float _t;
// silent all the debug infos (usefull only for small problems)
#define V(comands) ({})
// in case you want debug infos, this macro mange to print it properly.
// WARNING: LOCKS!!!!!
#ifndef V
pthread_mutex_t m_print ;
bool init = false;
#define V(comands) ({ if ( !init ) { init = true; pthread_mutex_init(&m_print, NULL); } \
			pthread_mutex_lock(&m_print); comands; pthread_mutex_unlock(&m_print); })
#endif
// still a dumper to leave as debug or to silent by redefining it. This one is used for times
#ifndef NO_TIMES
#define T(comands) ({comands;})
#endif 
#ifdef NO_TIMES
#define T(comands) ({})
#endif
// macro to eval the index.
#define I(i,j,cols) (i*cols+j)

// UTILITY FUNCTIONS 
/** TIC TOCK **
 * To manage the time, i am using two support functions (tic, tock) inspired to the following link
    https://stackoverflow.com/questions/13485266/how-to-have-matlab-tic-toc-in-c
 */
stack<clock_t> times; 
inline void tic() { times.push(clock()); }
inline void toc(const char* msg) {
    cout << msg << "(s):" << ((double)(clock() - times.top())) / CLOCKS_PER_SEC << endl;
    times.pop();
}



// simple abs
inline _t ABS(_t val){ return (val>=0)?val:-val; }
// determinate if two elements are equals with a tollerance of epsilon
inline bool almost(_t a, _t b, _t epsilon){
	if ( a == 0 ) return ABS(b) <= epsilon;
	if ( b == 0 ) return ABS(a) <= epsilon;
	else return ABS( a/b ) <= 1. + epsilon;
}
// determinate if the max_iterations were performed or the arrays are similar
bool endCondition(int i, int max_i, _t *x, _t *y, int n, _t epsilon){
	assert ( max_i>0 || epsilon >0 );
	if ( epsilon < 0 ) return i >= max_i ;
	if ( i >= max_i ) return true;
	for ( int k=0; k<n; k++ )
		if ( !almost( x[k], y[k], epsilon ) ) return false;  
	return true;
}
// allocate a matrix of nxm represented as an array
inline _t* create(int N, int M){
	_t *m = (_t*) malloc( sizeof ( _t ) * N * M );
	assert( m ) ;
	return m;	
}
// allocate and initialize a matrix to 0
inline _t* create_empty_vector(int n){
	_t *x = create(n,1);
	for(int i=0; i<n; i++) x[i] = 0;
	return x;
}
// evals p from A and b of size n in linear time
inline _t* create_p(int n, _t*A, _t*b){
	_t* p = create(n,1);
        for (int i=0; i<n; i++) p[i] = b[i] / A[I(i,i,n)];
	return p;
}
// evals J from A of size n*n in time n sqrt
inline _t* create_J(int n,_t*A){
	_t* J = create(n,n);
        for (int i=0; i<n; i++)
                for(int j=0; j<n; j++)
                        if ( i==j ) J[I(i,j,n)] = 0;
                        else J[I(i,j,n)] = -A[I(i,j,n)] / A[I(i,i,n)];
	return J;
}
/* since the diagonal will be inverted, it is checked if all of its elements are not null */
bool checkDiagonal( _t* A, int n ){
	for(int i=0; i<n; i++) if ( A[I(i,i,n)] == 0. ) return false;
	return true;
}
/******************* MATRIX GENERATOR ********************/
_t my_random(){
    	_t scale = 1000.;
    	_t max = 1024.*scale, min = -1024.*scale;
    	_t i = 0, d = 0;
    	i = rand() % (int)( max - min ) + min ;
    	d = i / scale;
	return d;
}
// creates a problem that is with diagonal predominance
void create_problem(int n, _t** A, _t** b){
	*A = create(n,n);
	for ( int i=0; i<n; i++ ){
		_t sum = 0.;
		for ( int j=0; j<n; j++ )
			sum += ABS( A[0][I(i,j,n)] = my_random() );
		if ( ABS( A[0][I(i,i,n)] ) <= sum - ABS(A[0][I(i,i,n)]) ) 
			A[0][I(i,i,n)] = sum;
	}
	*b = create(n,1);
	for(int i=0; i<n; i++) b[0][i] = my_random();
}
/********************************************************/

// do the multiplaction of the lines between i_ini and i_fin and store the result in dest (eventually summing sum)
void mul_range(_t* A, int na, int ma, _t*B, int nb, int mb, _t* dest, _t* optional_sum, int i_ini, int i_fin){
	assert ( ma == nb );
	for (int i=i_ini; i<=i_fin && i<na; i++)
		for ( int j=0; j<mb; j++ ){
			_t s = 0;
			for ( int k=0; k<ma; k++ )
				s+= A[I(i,k,ma)] * B[I(k,j,mb)];
			dest[I(i,j,mb)] = s + (optional_sum ? optional_sum[I(i,j,mb)] : 0 );
		}
}
// mult two matrixes in time n sqrt
void mul(_t* A, int na, int ma, _t*B, int nb, int mb, _t* dest, _t* optional_sum){
        mul_range(A,na,ma,B,nb,mb,dest,optional_sum, 0,na-1);
}


/****************************************************************/
/******************** LINEAR IMPLEMENTATION *********************/
/****************************************************************/
_t* linear( int n, _t *A, _t *b, int max_it, float epsilon ){
	T(tic(););
	assert ( checkDiagonal ( A, n ) );
	_t* p = create_p(n, A, b);
	_t* J = create_J(n, A);
	_t* x = create_empty_vector(n), *oldx = create(n,1);
	T(toc("\tInitialization"););
	T(tic(););
	int iteration = 0;
	do {
		iteration ++;
		// swaps the two arrays
		_t* t = x;
		x = oldx;
		oldx = t;
		
		mul(J, n,n, oldx, n,1, x, p);
	}while ( !endCondition(iteration, max_it, x, oldx, n, epsilon ) );
	T(toc("\tLoop"););
	T(cerr<<"\tIterations:"<<iteration<<endl;);
	T(tic();); 
	free( oldx );
	free( p );
	free( J );
	T(toc("\tCleaning"););
	return x;
}
/****************************************************************/
/******************** PARALLEL IMPLEMENTATION *******************/
/****************************************************************/
/* data structure, used as task for the farm as well, meant to store 
	useful infos of the workers */
struct worker_info{
	worker_info(){};
	worker_info(int index, int i, int f, int n, int w, int*counter, int*perform, _t** ptr_x, _t** ptr_oldx, pthread_mutex_t *m_comand, 
		pthread_mutex_t *m_control, pthread_cond_t *cond_comand, pthread_cond_t *cond_control, _t*J, _t*p):
		index(index), i(i), f(f), n(n), w(w), counter(counter), perform(perform),ptr_x(ptr_x), ptr_oldx(ptr_oldx), m_comand(m_comand), 
		m_control(m_control), cond_comand(cond_comand), cond_control(cond_control), J(J), p(p){};
	int index;
	int i, f;
	int n, w;
	int *counter, *perform;
	_t **ptr_x, **ptr_oldx;
	pthread_mutex_t *m_comand, *m_control;
	pthread_cond_t *cond_comand, *cond_control;
	_t *J, *p;
};
// more or less the same of the linaer while body but with locks!
void* worker(void *arg){
	worker_info *info = (worker_info*)arg;
	int p = 0;
	bool exit = false;
	V( int index = info->index; );
	V( cout << "Created thread " << index << ": taking care of [ "<< info->i << ", " << info->f << " ]" << endl ; );
        while ( !exit ) { 
		V( cout << index << ") locking comand"<< endl; );
	        // wait until new cicle of muls has to be performed
                pthread_mutex_lock(info->m_comand);
		V( cout << index << ") current perform is "<< *(info->perform) << "is it higher than "<< p << "?"<<endl; );
		if ( *(info->perform) < 0 ) exit = true;
                while( !exit && *(info->perform) <= p ){ 
			pthread_cond_wait(info->cond_comand, info->m_comand);
			if ( *(info->perform) < 0 ){ 
				exit = true;
				V( cout << index << ") required to exit" << endl; );
			}
			V( cout << index << ") current perform is "<< *(info->perform) << "is it higher than "<< p << "?"<<endl; );
		}
                pthread_mutex_unlock(info->m_comand);
		if ( exit ) break;
		V( cout << index << ") unlocking comand"<< endl; );
                p++;
                // mul
                mul_range(info->J, info->n,info->n, *(info->ptr_oldx), info->n, 1, *(info->ptr_x), info->p, info->i, info->f);
		V(cout << index << ") mul performed, locking "<< endl; );
                // send to dispatcher signal de end of job
                pthread_mutex_lock(info->m_control);
                *(info->counter) = 1 + *(info->counter);
		V(cout << index << ") counter incremented to " << *(info->counter) << endl; );
                if ( *(info->counter) == info->w )
			pthread_cond_signal(info->cond_control);
                pthread_mutex_unlock(info->m_control);
        }
	V( cout << index << ") shutting down" << endl; );
        return (void*)NULL;
};

_t* parallel( int w, int n, _t* A, _t *b, int max_it, float epsilon ) {
	T(tic(););
	assert( checkDiagonal ( A, n ) );
	assert( w > 0 );
	assert( n > w );
	V(cout<<"Initializing"<<endl;);
	pthread_t threads[ w ];
	worker_info tasks[w];

	int counter = 0, perform = 0;
	pthread_mutex_t m_control, m_comand ;
	pthread_cond_t cond_control, cond_comand ;
	pthread_mutex_init(&m_control, NULL);
	pthread_cond_init(&cond_control, NULL);
	pthread_mutex_init(&m_comand, NULL);
	pthread_cond_init(&cond_comand, NULL);
	
	V(cout<<"Creating oldx, x, p"<<endl;);
	_t * oldx = create(n,1), *x=create_empty_vector(n);
	// eval p
        _t* p = create_p(n,A,b);
        // eval J
	V(cout<<"creating J"<<endl;);
        _t* J = create_J(n,A);
       	// creating threads
	// worker_info(int i, int f, int n, int w, int*counter, int*perform, _t** ptr_x, _t** ptr_oldx, pthread_mutex_t *m_comand,
        //        pthread_mutex_t *m_control, pthread_cond_t *cond_comand, pthread_cond_t *cond_control, _t*J, _t*p): 
	V(cout<<"creating worker_info"<<endl;); 
	float portion = ((float)n)/(float)w;
	for ( int i=0 ; i<w; i++ ) {
		tasks[i] = worker_info( 
			i,
			portion * (float)i,  min(n, (int)(portion * (float) ( i + 1 )))- 1,
			n, w, &counter, &perform, &x, &oldx,
			&m_comand, &m_control, &cond_comand, &cond_control,
			J,p); 
		int r = pthread_create( &threads[i], NULL, worker, &(tasks[i]) );
		assert ( !r );
	}
	V(cout << "Initialization done" << endl ; );
	T(toc("\tInitialization"););
	T(tic(););
	// loop
        int iteration = perform = counter = 0;
        do {
		iteration++;
		V(cout << "Loop cycle "<< iteration << endl; );
		_t* t = x;
		x = oldx;
		oldx = t;
                pthread_mutex_lock(&m_comand);
		perform++;
                pthread_cond_broadcast(&cond_comand);
                pthread_mutex_unlock(&m_comand);
		// wait for task to be computed
		pthread_mutex_lock(&m_control);
		while ( counter != w ) pthread_cond_wait(&cond_control, &m_control);
		counter = 0;
		pthread_cond_broadcast(&cond_control);
		pthread_mutex_unlock(&m_control);
		// x at the end of the day
		V( cout << "Iteration "<< iteration << ": X[ "; for (int i = 0 ; i<n; i++ ) cout << x[i] << ((i==n-1)?" ]": ", "); cout << endl; );
        }while ( !endCondition(iteration, max_it, x, oldx, n, epsilon ) );
	T(toc("\tLoop"););
	T(cerr<<"\tIterations:"<<iteration<<endl;);
	T(tic(););
	V(cout << "Ended algorithm at iteration "<< iteration << ". telling the fact to threads" << endl;);
	pthread_mutex_lock(&m_comand);
	perform = -1;
	pthread_cond_broadcast(&cond_comand);
	pthread_mutex_unlock(&m_comand);
	V(cout << "Ending routine" << endl; );
	// waiting the end 
	for ( int i=0; i<w; i++ ){
		int rc = pthread_join( threads[i], NULL );
		assert ( !rc ); 
	}
	V(cout << "End."<< endl;);
	free(oldx);
	free(p);
	free(J);
	T(toc("\tCleanning"););
	return x;
}

// ----------------------------------
/** FAST FLOW IMPLEMENTATION !!! **/
// ----------------------------------

struct farm_worker_t:ff_node_t<worker_info,worker_info>{
	ParallelFor *pr = NULL;
	farm_worker_t( int f ) {	
		if ( f > 1 ) pr = new ParallelFor(f);
		else if ( f != 1 ) pr = new ParallelFor();
	}
	~farm_worker_t(){ if ( pr ) delete( pr ) ; }		
	worker_info* svc(worker_info* arg){
		if ( ! arg ) return (worker_info*) GO_ON;
		worker_info *info = (worker_info*) arg;
		V(cout << "W: " << info->index << endl; );
		if ( pr == NULL ) 
			mul_range(info->J, info->n,info->n, *(info->ptr_oldx), info->n, 1, *(info->ptr_x), info->p, info->i, info->f);
		else 
			pr->parallel_for(info->i, info->f+1, [info](const long i){
					mul_range(info->J, info->n, info->n, *(info->ptr_oldx), info->n, 1, *(info->ptr_x), info->p, i,i);
				});			
		return arg;
	}
};
struct emitter_t:public ff_node_t<worker_info>{
	int w;
	worker_info *infos;
	_t **ptr_x, **ptr_oldx;
	emitter_t(int w, worker_info *infos, _t** px, _t** pox):
		w(w), infos(infos), ptr_x(px), ptr_oldx(pox){};
	worker_info* svc( worker_info* info ) {
		_t* t = *ptr_x;
                *ptr_x = *ptr_oldx;
                *ptr_oldx = t;
		for ( int i=0 ; i<w; i++ ){ 
			V(cout<<"E: "<< (int)i << endl;);
			ff_send_out ( &(infos[i]) );	
		}
		if ( info == (worker_info*) EOS ) return info;
		else return (worker_info*) GO_ON;	
	};
};
struct collector_t: public ff_node_t<worker_info>{
	int w, max_it,n;
	_t **ptr_x;
	_t **ptr_oldx;
	_t epsilon;
	int counter, iteration;
	collector_t(int w, int n, _t **ptr_x, _t **ptr_oldx, int max_it, _t eps): 
		w(w), max_it(max_it), n(n), ptr_x(ptr_x), ptr_oldx(ptr_oldx), epsilon(eps){
		counter = iteration = 0;
		assert( ptr_x && ptr_oldx );
	};
	worker_info* svc( worker_info* info ){
		worker_info * result = (worker_info*) GO_ON;
		V( cout << "C " << info << endl; );
		counter++;
		if ( counter == w ) {
			counter = 0;
			iteration ++;
			V(cout << "C: All arrived "<< endl;);
			if ( endCondition(iteration, max_it, *ptr_x, *ptr_oldx, n, epsilon ) ){
				result = (worker_info*) EOS;	
				T(cerr<<"\tIterations:"<<iteration<<endl;);
			}
			else result = info;	
		}
		return result;
	};
};

_t* farm( int w, int f, int n, _t* A, _t *b, int max_it, _t epsilon ) {
	T(tic(););
	assert( checkDiagonal ( A, n ) );
        assert( w > 0 && f > 0 );
        assert( n > w );
	// initialiating the problem
	_t * oldx = create(n,1), *x=create_empty_vector(n);
        _t* p = create_p(n,A,b);
        _t* J = create_J(n,A);
	worker_info tasks[w];
	float portion = ((float)n)/(float)w;
        for ( int i=0 ; i<w; i++ ) 
                tasks[i] = worker_info(
                        i,
                        portion * (float)i,  min(n, (int)(portion * (float) ( i + 1 )))- 1,
                        n, w, NULL,NULL, &x, &oldx,NULL,NULL,NULL,NULL,J,p);
	// creating the farm
        emitter_t emitter(w,tasks, &x, &oldx);
	collector_t collector( w, n, &x, &oldx, max_it, epsilon );
        vector< ff_node* >workers;
        for ( int i=0; i<w; i++ ) workers.push_back( new farm_worker_t(f) );
        ff_farm<> farm(workers);
        farm.add_emitter( &emitter ) ;
        farm.remove_collector();
	farm.add_collector( &collector );
	farm.setMultiInput();
        farm.wrap_around();
	T(toc("\tInitialization"););
	T(tic(););
	// running 
	if ( farm.run_and_wait_end() < 0 ) error("pipe's run and wait");
	T(toc("\tLoop"););
	T(tic(););
	#ifdef TRACE_FASTFLOW
	T( cerr<<"Spent(ms): "<<farm.ffTime()<<endl; );
	T( cerr<<"Farm stats: "<< endl; farm.ffStats( cerr ); cerr<<endl; );
	#endif
	// shutting down
	for ( int i=0; i<w; i++ ) delete( workers[i] );
	free( p );
	free( J );
	free( oldx );
	T(toc("\tCleanning"););
	return x;
}
/******************** PRINTING THE USAGE ****************************/
void printUsage(const char program_name[]){
        cerr<<"Usage : "<<program_name<<" [ "<<endl
	<< "\t\t| -w val\tnum of workers "<<endl
	<< "\t\t| -t val\tinternal paralelism for farm"<< endl
	<< "\t\t| -n val\tsize of the problem to generate (in case input_file is unspecified)" << endl
	<< "\t\t| -l \t\t(run the linear version) " << endl
	<< "\t\t| -p \t\t(run the plain thread version)" << endl
	<< "\t\t| -f \t\t(run the farmed version FASTFLOW) " << endl
	<< "\t\t| -m val\tmax iteration (-1 to have no limit)" << endl
	<< "\t\t| -e valf\terror tolareted between two consegutive val (-1 to have no tollerance)" << endl
	<< "\t\t| -s \t\tsilent the output" << endl
	<< "\t\t| -r \t\tdo srand on time(NULL) " << endl
	<< "\t\t| -h \t\tprint this message and exit" << endl
	<< "\t] " << endl << " \t[input_file] \t(In this case -n can't be used)"<<endl
                <<"\tWhere input_file has the following format:"<<endl
                <<"\t\tN"<<endl
                <<"\t\ta11  a12  ... a1N"<<endl
                <<"\t\t...  ...  ... ..."<<endl
                <<"\t\taN1  aN2  ... aNN"<<endl
                <<"\t\tb1   b2   ... bN"<<endl
                <<endl;
}
/* ---------------------------------
 -----------------------------------
           MAIN 
 ----------------------------------
----------------------------------- */

int main(int argc, char** argv){
	srand( 0 );
	int opt ;
	int n = -1, w=1, t=1;
	bool _l = false, _p = false, _f = false, cmd[3]{false, false, false}, silent = false;
	int max_it = 100;
	float epsilon = -1;
	_t *A, *b, *r;
	while ( (opt=getopt(argc,argv,"rshlpfn:m:e:w:t:") ) != -1 )
		switch ( opt ) {
			case 'h': printUsage(argv[0]); exit(EXIT_SUCCESS); break;
			case 'n': n = atoi ( optarg ) ; break;
			case 'l': cmd[0] = _l = true; break;
			case 'p': cmd[1] = _p = true; break;
			case 'f': cmd[2] = _f = true; break;
			case 'm': max_it = atoi(optarg) ; break;
			case 'e': epsilon= atof(optarg) ; break;
			case 'w': w = atoi(optarg); break;
			case 't': t = atoi(optarg); break;
			case 's': silent = true; break;
			case 'r': srand(time(0)); break;
			default : 
				printUsage( argv[0] );
				exit ( EXIT_FAILURE );
		}
	char * file = ( optind >= argc )?NULL: argv[optind];
	if ( file ) {
		T(tic(););
		assert ( n == -1 );
		ifstream in(file);
		assert ( in ) ;
		in >> n; 
		A = create(n,n);
		b = create(n,1);
		assert( n > 0 );
		for ( int i=0 ; i<n*n ; i++ ) in >> A[i];
		for ( int i=0; i< n; i++ ) in >> b[i]; 
		in.close();
		T(toc("Problem_reading"););
	}else{ 
		if ( n < 1 ){
			printUsage(argv[0]);
			exit( EXIT_FAILURE );
		}
		T(tic(););
		A = create(n,n);
		b = create(n,1);
		create_problem( n , &A, &b );
		T(toc("Problem_creation"););
	} 
	assert ( ( _l || _p || _f ) );
	if ( ! silent ) {
		cout << "RUNNING: " << endl
			<< "\tn " << n << endl
			<< "\tw " << w << endl
			<< "\tt " << t << endl
			<< "\tm " << max_it << endl
			<< "\te " << epsilon<< endl
			<< "\tr " << r << endl
			<< "\tlinear " << _l << endl
			<< "\tplain thread " << _p << endl
			<< "\tfarm " << _f << endl
			<< "\tff's MAX_NUMTHREADS " << MAX_NUM_THREADS << endl
			<< endl;
	}
	for ( int i=0 ; i<3 ; i++ )
		if ( cmd [i] ) {
			const char * name = ( (i==0)?"linear": ( (i==1)?"thread": "farmed" )  );
			if ( !silent ) cout << name << endl;
			else T(cout<<name<<endl;); 
			T(tic(););
			if ( i == 0 ) r = linear ( n, A, b, max_it, epsilon ) ;
			else if ( i== 1)r=parallel(w,n,A,b, max_it, epsilon) ;
			else r = farm ( w, t, n, A, b, max_it, epsilon );
			T(toc("\tEnlapsed time"););
			#ifndef NO_RESULT
			if ( !silent ){
				cout << "\tresult : ";
				for ( int i=0; i<n; i++ ) cout << r[i] << " " ; 
				cout << endl;
			}
			#endif
			free( r );
		}
	free( A );
	free( b );
	return 0;
}
