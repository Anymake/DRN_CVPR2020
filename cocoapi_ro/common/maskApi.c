/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#include "maskApi.h"
#include <math.h>
#include <stdlib.h>

uint umin( uint a, uint b ) { return (a<b) ? a : b; }
uint umax( uint a, uint b ) { return (a>b) ? a : b; }

void OverlapSub (double *rbox1, double *rbox2, double *area, int *crowd)
{
    double xcenter1 = rbox1[0];
    double ycenter1 = rbox1[1];
    double width1 = rbox1[2];
    double height1 = rbox1[3];
    double angle1 = rbox1[4];
    double xcenter2 = rbox2[0];
    double ycenter2 = rbox2[1];
    double width2 = rbox2[2];
    double height2 = rbox2[3];
    double angle2 = rbox2[4];
    //for(int i=0;i<5;i++) cout<<rbox1[i]<<" ";
    //cout<<endl;
    //for(int i=0;i<5;i++) cout<<rbox2[i]<<" ";
    //cout<<endl;
	angle1 = -angle1;
	angle2 = -angle2;
	double angled = angle2 - angle1;
	angled *= (double)3.14159265/180;
	angle1 *= (double)3.14159265/180;
	//ofstream fout("Output.txt");
	area[0] = 0;
	double hw1 = width1 / 2;
	double hh1 = height1 /2;
	double hw2 = width2 / 2;
	double hh2 = height2 /2;
	double xcenterd = xcenter2 - xcenter1;
	double ycenterd = ycenter2 - ycenter1;
	double tmp = xcenterd * cosf(angle1) + ycenterd * sinf(angle1);
	ycenterd = -xcenterd * sinf(angle1) + ycenterd * cosf(angle1);
	xcenterd = tmp;
	double max_width_height1 = width1 > height1? width1 : height1;
	double max_width_height2 = width2 > height2? width2 : height2;
	if (sqrt(xcenterd * xcenterd + ycenterd * ycenterd) >
		(max_width_height1 + max_width_height2) * 1.414214/2)
	{
		area[0] = 0;
		//fout<<endl<<"AREA = 0"<<endl;
		//fout.close();
		return;
	}
	if (fabs(sin(angled)) < 1e-3)
	{
		if (fabs(xcenterd) > (hw1 + hw2) || fabs(ycenterd) > (hh1 + hh2))
		{
			area[0] = 0;
			//fout<<endl<<"AREA = 0"<<endl;
			//fout.close();
			return;
		}
		else
		{
			double x_min_inter = -hw1 > (xcenterd - hw2)? -hw1 : (xcenterd - hw2);
			double x_max_inter = hw1 < (xcenterd + hw2)? hw1 : (xcenterd + hw2);
			double y_min_inter = -hh1 > (ycenterd - hh2)? -hh1 : (ycenterd - hh2);
			double y_max_inter = hh1 < (ycenterd + hh2)? hh1 : (ycenterd + hh2);
			const double inter_width = x_max_inter - x_min_inter;
			const double inter_height = y_max_inter - y_min_inter;
			const double inter_size = inter_width * inter_height;
			area[0] = inter_size;
            area[0] = area[0] / (width1 * height1 + width2 * height2 - area[0]);
			//LOG(INFO)<<"AREA = "<<area;
			//fout.close();
			return;
		}
	}
	if (fabs(cos(angled)) < 1e-3)
	{
		double x_min_inter = -hw1 > (xcenterd - hh2)? -hw1 : (xcenterd - hh2);
		double x_max_inter = hw1 < (xcenterd + hh2)? hw1 : (xcenterd + hh2);
		double y_min_inter = -hh1 > (ycenterd - hw2)? -hh1 : (ycenterd - hw2);
		double y_max_inter = hh1 < (ycenterd + hw2)? hh1 : (ycenterd + hw2);
		const double inter_width = x_max_inter - x_min_inter;
		const double inter_height = y_max_inter - y_min_inter;
		const double inter_size = inter_width * inter_height;
		area[0] = inter_size;
        area[0] = area[0] / (width1 * height1 + width2 * height2 - area[0]);
		//fout<<endl<<"AREA = "<<area<<endl;
		//fout.close();
		return;
	}

	double cos_angled = cosf(angled);
	double sin_angled = sinf(angled);
	double cos_angled_hw1 = cos_angled * hw1;
	double sin_angled_hw1 = sin_angled * hw1;
	double cos_angled_hh1 = cos_angled * hh1;
	double sin_angled_hh1 = sin_angled * hh1;
	double cos_angled_hw2 = cos_angled * hw2;
	double sin_angled_hw2 = sin_angled * hw2;
	double cos_angled_hh2 = cos_angled * hh2;
	double sin_angled_hh2 = sin_angled * hh2;

	// point20: (w/2, h/2)
	double point2x[4], point2y[4];
	point2x[0] = xcenterd + cos_angled_hw2 - sin_angled_hh2;
	point2y[0] = ycenterd + sin_angled_hw2 + cos_angled_hh2;
	// point21: (-w/2, h/2)
	point2x[1] = xcenterd - cos_angled_hw2 - sin_angled_hh2;
	point2y[1] = ycenterd - sin_angled_hw2 + cos_angled_hh2;
	// point22: (-w/2, -h/2)
	point2x[2] = xcenterd - cos_angled_hw2 + sin_angled_hh2;
	point2y[2] = ycenterd - sin_angled_hw2 - cos_angled_hh2;
	// point23: (w/2, -h/2)
	point2x[3] = xcenterd + cos_angled_hw2 + sin_angled_hh2;
	point2y[3] = ycenterd + sin_angled_hw2 - cos_angled_hh2;

	double pcenter_x = 0, pcenter_y = 0;
	int count = 0;

	// determine the inner point
	int inner_side2[4][4], inner2[4];
	for(int i = 0; i < 4; i++)
	{
		inner_side2[i][0] = point2y[i] < hh1;
		inner_side2[i][1] = point2x[i] > -hw1;
		inner_side2[i][2] = point2y[i] > -hh1;
		inner_side2[i][3] = point2x[i] < hw1;
		inner2[i] = inner_side2[i][0] & inner_side2[i][1] & inner_side2[i][2] & inner_side2[i][3];
		if (inner2[i]) { pcenter_x += point2x[i]; pcenter_y += point2y[i]; count++;}
	}

	//similar operating for rbox1: angled -> -angled, xcenterd -> -xcenterd, ycenterd -> -ycenterd
	// point10: (w/2, h/2)
	double xcenterd_hat = - xcenterd * cos_angled - ycenterd * sin_angled;
	double ycenterd_hat = xcenterd * sin_angled - ycenterd * cos_angled;
	double point1x[4], point1y[4];

	point1x[0] = xcenterd_hat + cos_angled_hw1 + sin_angled_hh1;
	point1y[0] = ycenterd_hat - sin_angled_hw1 + cos_angled_hh1;
	// point21: (-w/2, h/2)
	point1x[1] = xcenterd_hat - cos_angled_hw1 + sin_angled_hh1;
	point1y[1] = ycenterd_hat + sin_angled_hw1 + cos_angled_hh1;
	// point22: (-w/2, -h/2)
	point1x[2] = xcenterd_hat - cos_angled_hw1 - sin_angled_hh1;
	point1y[2] = ycenterd_hat + sin_angled_hw1 - cos_angled_hh1;
	// point23: (w/2, -h/2)
	point1x[3] = xcenterd_hat + cos_angled_hw1 - sin_angled_hh1;
	point1y[3] = ycenterd_hat - sin_angled_hw1 - cos_angled_hh1;

	// determine the inner point
	// determine the inner point
	int inner_side1[4][4], inner1[4];
	for(int i = 0; i < 4; i++)
	{
		inner_side1[i][0] = point1y[i] < hh2;
		inner_side1[i][1] = point1x[i] > -hw2;
		inner_side1[i][2] = point1y[i] > -hh2;
		inner_side1[i][3] = point1x[i] < hw2;
		inner1[i] = inner_side1[i][0] & inner_side1[i][1] & inner_side1[i][2] & inner_side1[i][3];
	}
	point1x[0] = hw1;
	point1y[0] = hh1;
	// point21: (-w/2, h/2)
	point1x[1] = -hw1;
	point1y[1] = hh1;
	// point22: (-w/2, -h/2)
	point1x[2] = -hw1;
	point1y[2] = -hh1;
	// point23: (w/2, -h/2)
	point1x[3] = hw1;
	point1y[3] = -hh1;
	if (inner1[0]) { pcenter_x += hw1; pcenter_y += hh1; count++;}
	if (inner1[1]) { pcenter_x -= hw1; pcenter_y += hh1; count++;}
	if (inner1[2]) { pcenter_x -= hw1; pcenter_y -= hh1; count++;}
	if (inner1[3]) { pcenter_x += hw1; pcenter_y -= hh1; count++;}
	//find cross_points
	Line line1[4], line2[4];
	line1[0].p1 = 0; line1[0].p2 = 1;
	line1[1].p1 = 1; line1[1].p2 = 2;
	line1[2].p1 = 2; line1[2].p2 = 3;
	line1[3].p1 = 3; line1[3].p2 = 0;
	line2[0].p1 = 0; line2[0].p2 = 1;
	line2[1].p1 = 1; line2[1].p2 = 2;
	line2[2].p1 = 2; line2[2].p2 = 3;
	line2[3].p1 = 3; line2[3].p2 = 0;
	double pointc_x[4][4], pointc_y[4][4];
	for (int i = 0; i < 4; i++)
	{
		int index1 = line1[i].p1;
		int index2 = line1[i].p2;
		line1[i].crossnum = 0;
		if (inner1[index1] && inner1[index2])
		{
			if (i == 0 || i == 2) line1[i].length = width1;
			else line1[i].length = height1;
			line1[i].crossnum = -1;
			continue;
		}
		if (inner1[index1])
		{
			line1[i].crossnum ++;
			line1[i].d[0][0] = index1;
			line1[i].d[0][1] = -1;
			continue;
		}
		if (inner1[index2])
		{
			line1[i].crossnum ++;
			line1[i].d[0][0] = index2;
			line1[i].d[0][1] = -1;
			continue;
		}
	}
	for (int i = 0; i < 4; i++)
	{
		int index1 = line2[i].p1;
		double x1 = point2x[index1];
		double y1 = point2y[index1];
		int index2 = line2[i].p2;
		double x2 = point2x[index2];
		double y2 = point2y[index2];
		line2[i].crossnum = 0;
		if (inner2[index1] && inner2[index2])
		{
			if (i == 0 || i == 2) line2[i].length = width2;
			else line2[i].length = height1;
			line2[i].crossnum = -1;
			continue;
		}
		if (inner2[index1])
		{
			line2[i].crossnum ++;
			line2[i].d[0][0] = index1;
			line2[i].d[0][1] = -1;
		}
		else if (inner2[index2])
		{
			line2[i].crossnum ++;
			line2[i].d[0][0] = index2;
			line2[i].d[0][1] = -1;
		}
		double tmp1 = (y1*x2 - y2*x1) / (y1 - y2);
		double tmp2 = (x1 - x2) / (y1 - y2);
        //cout<<"tmp"<<" "<<i<<" "<<tmp1<<" "<<tmp2<<endl;
        //cout<<"x1="<<x1<<" x2="<<x2<<endl;
        //cout<<"y1="<<y1<<" y2="<<y2<<endl;
		double tmp3 = (x1*y2 - x2*y1) / (x1 - x2);
		double tmp4 = 1/tmp2 * hw1;
		tmp2 *= hh1;
		for (int j = 0; j < 4; j++)
		{
			int index3 = line1[j].p1;
			int index4 = line1[j].p2;
			if ((inner_side2[index1][j] != inner_side2[index2][j])
				&& (inner_side1[index3][i] != inner_side1[index4][i]))
			{
				switch (j)
				{
				case 0:
					pointc_x[i][j] = tmp1 + tmp2;
					pointc_y[i][j] = hh1;
					break;
				case 1:
					pointc_y[i][j] = tmp3 - tmp4;
					pointc_x[i][j] = -hw1;
					break;
				case 2:
					pointc_x[i][j] = tmp1 - tmp2;
					pointc_y[i][j] = -hh1;
					break;
				case 3:
					pointc_y[i][j] = tmp3 + tmp4;
					pointc_x[i][j] = hw1;
					break;
				default:
					break;
				}
				line1[j].d[line1[j].crossnum][0] = i;
				line1[j].d[line1[j].crossnum ++][1] = j;
				line2[i].d[line2[i].crossnum][0] = i;
				line2[i].d[line2[i].crossnum ++][1] = j;
				pcenter_x += pointc_x[i][j];
				pcenter_y += pointc_y[i][j];
				count ++;
                //cout<<"pointc:"<<i<<" "<<j<<" "<<pointc_x[i][j]<<" "<<pointc_y[i][j]<<endl;
			}
		}
	}
	pcenter_x /= (double)count;
	pcenter_y /= (double)count;
	double pcenter_x_hat, pcenter_y_hat;
	pcenter_x_hat = pcenter_x - xcenterd;
	pcenter_y_hat = pcenter_y - ycenterd;
	tmp = cos_angled * pcenter_x_hat + sin_angled * pcenter_y_hat;
	pcenter_y_hat = -sin_angled * pcenter_x_hat + cos_angled * pcenter_y_hat;
	pcenter_x_hat = tmp;

	for (int i = 0; i < 4; i++)
	{
		if (line1[i].crossnum > 0)
		{
			if (line1[i].d[0][1] == -1)
			{
				if (i==0 || i==2)
					line1[i].length = fabs(point1x[line1[i].d[0][0]] - pointc_x[line1[i].d[1][0]][line1[i].d[1][1]]);
				else
					line1[i].length = fabs(point1y[line1[i].d[0][0]] - pointc_y[line1[i].d[1][0]][line1[i].d[1][1]]);
			}
			else
			{
				if (i==0 || i==2)
					line1[i].length = fabs(pointc_x[line1[i].d[0][0]][line1[i].d[0][1]] - pointc_x[line1[i].d[1][0]][line1[i].d[1][1]]);
				else
					line1[i].length = fabs(pointc_y[line1[i].d[0][0]][line1[i].d[0][1]] - pointc_y[line1[i].d[1][0]][line1[i].d[1][1]]);
			}
		}
		if (line2[i].crossnum >0)
		{
			if (line2[i].d[0][1] == -1)
				line2[i].length = fabs(point2x[line2[i].d[0][0]] - pointc_x[line2[i].d[1][0]][line2[i].d[1][1]]);
			else
				line2[i].length = fabs(pointc_x[line2[i].d[0][0]][line2[i].d[0][1]] - pointc_x[line2[i].d[1][0]][line2[i].d[1][1]]);
			if(i == 0 || i == 2) line2[i].length *= width2 / fabs(point2x[line2[i].p1] - point2x[line2[i].p2]);
			else line2[i].length *= height2 / fabs(point2x[line2[i].p1] - point2x[line2[i].p2]);
		}
	}

	double dis1[4], dis2[4];
	dis1[0] = fabs(pcenter_y - hh1);
	dis1[1] = fabs(pcenter_x + hw1);
	dis1[2] = fabs(pcenter_y + hh1);
	dis1[3] = fabs(pcenter_x - hw1);
	dis2[0] = fabs(pcenter_y_hat - hh2);
	dis2[1] = fabs(pcenter_x_hat + hw2);
	dis2[2] = fabs(pcenter_y_hat + hh2);
	dis2[3] = fabs(pcenter_x_hat - hw2);
	for (int i=0; i < 4; i++)
	{
        //cout<<"line1["<<i<<"].crossnum="<<line1[i].crossnum<<endl;
		if (line1[i].crossnum != 0)
            //cout<<"line1 "<<dis1[i]<<" "<<line1[i].length<<endl;
			area[0] += dis1[i] * line1[i].length;
		if (line2[i].crossnum != 0)
            //cout<<"line2 "<<dis2[i]<<" "<<line2[i].length<<endl;
			area[0] += dis2[i] * line2[i].length;
	}
	area[0] /= 2;
    //cout<<area[0]<<endl;
//    area[0] = area[0] / (width1 * height1 + width2 * height2 - area[0]);
    double u = crowd ? width1 * height1 : width1 * height1 + width2 * height2 - area[0];
    area[0] = area[0] / u;
}


void rleInit( RLE *R, siz h, siz w, siz m, uint *cnts ) {
  R->h=h; R->w=w; R->m=m; R->cnts=(m==0)?0:malloc(sizeof(uint)*m);
  siz j; if(cnts) for(j=0; j<m; j++) R->cnts[j]=cnts[j];
}

void rleFree( RLE *R ) {
  free(R->cnts); R->cnts=0;
}

void rlesInit( RLE **R, siz n ) {
  siz i; *R = (RLE*) malloc(sizeof(RLE)*n);
  for(i=0; i<n; i++) rleInit((*R)+i,0,0,0,0);
}

void rlesFree( RLE **R, siz n ) {
  siz i; for(i=0; i<n; i++) rleFree((*R)+i); free(*R); *R=0;
}

void rleEncode( RLE *R, const byte *M, siz h, siz w, siz n ) {
  siz i, j, k, a=w*h; uint c, *cnts; byte p;
  cnts = malloc(sizeof(uint)*(a+1));
  for(i=0; i<n; i++) {
    const byte *T=M+a*i; k=0; p=0; c=0;
    for(j=0; j<a; j++) { if(T[j]!=p) { cnts[k++]=c; c=0; p=T[j]; } c++; }
    cnts[k++]=c; rleInit(R+i,h,w,k,cnts);
  }
  free(cnts);
}

void rleDecode( const RLE *R, byte *M, siz n ) {
  siz i, j, k; for( i=0; i<n; i++ ) {
    byte v=0; for( j=0; j<R[i].m; j++ ) {
      for( k=0; k<R[i].cnts[j]; k++ ) *(M++)=v; v=!v; }}
}

void rleMerge( const RLE *R, RLE *M, siz n, int intersect ) {
  uint *cnts, c, ca, cb, cc, ct; int v, va, vb, vp;
  siz i, a, b, h=R[0].h, w=R[0].w, m=R[0].m; RLE A, B;
  if(n==0) { rleInit(M,0,0,0,0); return; }
  if(n==1) { rleInit(M,h,w,m,R[0].cnts); return; }
  cnts = malloc(sizeof(uint)*(h*w+1));
  for( a=0; a<m; a++ ) cnts[a]=R[0].cnts[a];
  for( i=1; i<n; i++ ) {
    B=R[i]; if(B.h!=h||B.w!=w) { h=w=m=0; break; }
    rleInit(&A,h,w,m,cnts); ca=A.cnts[0]; cb=B.cnts[0];
    v=va=vb=0; m=0; a=b=1; cc=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); cc+=c; ct=0;
      ca-=c; if(!ca && a<A.m) { ca=A.cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<B.m) { cb=B.cnts[b++]; vb=!vb; } ct+=cb;
      vp=v; if(intersect) v=va&&vb; else v=va||vb;
      if( v!=vp||ct==0 ) { cnts[m++]=cc; cc=0; }
    }
    rleFree(&A);
  }
  rleInit(M,h,w,m,cnts); free(cnts);
}

void rleArea( const RLE *R, siz n, uint *a ) {
  siz i, j; for( i=0; i<n; i++ ) {
    a[i]=0; for( j=1; j<R[i].m; j+=2 ) a[i]+=R[i].cnts[j]; }
}

void rleIou( RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o ) {
  siz g, d; BB db, gb; int crowd;
  db=malloc(sizeof(double)*m*4); rleToBbox(dt,db,m);
  gb=malloc(sizeof(double)*n*4); rleToBbox(gt,gb,n);
  bbIou(db,gb,m,n,iscrowd,o); free(db); free(gb);
  for( g=0; g<n; g++ ) for( d=0; d<m; d++ ) if(o[g*m+d]>0) {
    crowd=iscrowd!=NULL && iscrowd[g];
    if(dt[d].h!=gt[g].h || dt[d].w!=gt[g].w) { o[g*m+d]=-1; continue; }
    siz ka, kb, a, b; uint c, ca, cb, ct, i, u; int va, vb;
    ca=dt[d].cnts[0]; ka=dt[d].m; va=vb=0;
    cb=gt[g].cnts[0]; kb=gt[g].m; a=b=1; i=u=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); if(va||vb) { u+=c; if(va&&vb) i+=c; } ct=0;
      ca-=c; if(!ca && a<ka) { ca=dt[d].cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<kb) { cb=gt[g].cnts[b++]; vb=!vb; } ct+=cb;
    }
    if(i==0) u=1; else if(crowd) rleArea(dt+d,1,&u);
    o[g*m+d] = (double)i/(double)u;
  }
}

void rleNms( RLE *dt, siz n, uint *keep, double thr ) {
  siz i, j; double u;
  for( i=0; i<n; i++ ) keep[i]=1;
  for( i=0; i<n; i++ ) if(keep[i]) {
    for( j=i+1; j<n; j++ ) if(keep[j]) {
      rleIou(dt+i,dt+j,1,1,0,&u);
      if(u>thr) keep[j]=0;
    }
  }
}

void bbIou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o ) {
  double h, w, i, u, ga, da; siz g, d; int crowd;
  for( g=0; g<n; g++ ) {
    BB G=gt+g*4; ga=G[2]*G[3]; crowd=iscrowd!=NULL && iscrowd[g];
    for( d=0; d<m; d++ ) {
      BB D=dt+d*4; da=D[2]*D[3]; o[g*m+d]=0;
      w=fmin(D[2]+D[0],G[2]+G[0])-fmax(D[0],G[0]); if(w<=0) continue;
      h=fmin(D[3]+D[1],G[3]+G[1])-fmax(D[1],G[1]); if(h<=0) continue;
      i=w*h; u = crowd ? da : da+ga-i; o[g*m+d]=i/u;
    }
  }
}

void rbbIou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o ) {
  siz g, d; int crowd;
  for( g=0; g<n; g++ ) {
    BB G=gt+g*5;
    crowd=iscrowd!=NULL && iscrowd[g];
    for( d=0; d<m; d++ ) {
      BB D=dt+d*5;
      o[g*m+d]=0;
      OverlapSub(G, D, &o[g*m+d], crowd);
    }
  }
}

void bbNms( BB dt, siz n, uint *keep, double thr ) {
  siz i, j; double u;
  for( i=0; i<n; i++ ) keep[i]=1;
  for( i=0; i<n; i++ ) if(keep[i]) {
    for( j=i+1; j<n; j++ ) if(keep[j]) {
      bbIou(dt+i*4,dt+j*4,1,1,0,&u);
      if(u>thr) keep[j]=0;
    }
  }
}

void rleToBbox( const RLE *R, BB bb, siz n ) {
  siz i; for( i=0; i<n; i++ ) {
    uint h, w, x, y, xs, ys, xe, ye, xp, cc, t; siz j, m;
    h=(uint)R[i].h; w=(uint)R[i].w; m=R[i].m;
    m=((siz)(m/2))*2; xs=w; ys=h; xe=ye=0; cc=0;
    if(m==0) { bb[4*i+0]=bb[4*i+1]=bb[4*i+2]=bb[4*i+3]=0; continue; }
    for( j=0; j<m; j++ ) {
      cc+=R[i].cnts[j]; t=cc-j%2; y=t%h; x=(t-y)/h;
      if(j%2==0) xp=x; else if(xp<x) { ys=0; ye=h-1; }
      xs=umin(xs,x); xe=umax(xe,x); ys=umin(ys,y); ye=umax(ye,y);
    }
    bb[4*i+0]=xs; bb[4*i+2]=xe-xs+1;
    bb[4*i+1]=ys; bb[4*i+3]=ye-ys+1;
  }
}

void rleFrBbox( RLE *R, const BB bb, siz h, siz w, siz n ) {
  siz i; for( i=0; i<n; i++ ) {
    double xs=bb[4*i+0], xe=xs+bb[4*i+2];
    double ys=bb[4*i+1], ye=ys+bb[4*i+3];
    double xy[8] = {xs,ys,xs,ye,xe,ye,xe,ys};
    rleFrPoly( R+i, xy, 4, h, w );
  }
}

int uintCompare(const void *a, const void *b) {
  uint c=*((uint*)a), d=*((uint*)b); return c>d?1:c<d?-1:0;
}

void rleFrPoly( RLE *R, const double *xy, siz k, siz h, siz w ) {
  /* upsample and get discrete points densely along entire boundary */
  siz j, m=0; double scale=5; int *x, *y, *u, *v; uint *a, *b;
  x=malloc(sizeof(int)*(k+1)); y=malloc(sizeof(int)*(k+1));
  for(j=0; j<k; j++) x[j]=(int)(scale*xy[j*2+0]+.5); x[k]=x[0];
  for(j=0; j<k; j++) y[j]=(int)(scale*xy[j*2+1]+.5); y[k]=y[0];
  for(j=0; j<k; j++) m+=umax(abs(x[j]-x[j+1]),abs(y[j]-y[j+1]))+1;
  u=malloc(sizeof(int)*m); v=malloc(sizeof(int)*m); m=0;
  for( j=0; j<k; j++ ) {
    int xs=x[j], xe=x[j+1], ys=y[j], ye=y[j+1], dx, dy, t, d;
    int flip; double s; dx=abs(xe-xs); dy=abs(ys-ye);
    flip = (dx>=dy && xs>xe) || (dx<dy && ys>ye);
    if(flip) { t=xs; xs=xe; xe=t; t=ys; ys=ye; ye=t; }
    s = dx>=dy ? (double)(ye-ys)/dx : (double)(xe-xs)/dy;
    if(dx>=dy) for( d=0; d<=dx; d++ ) {
      t=flip?dx-d:d; u[m]=t+xs; v[m]=(int)(ys+s*t+.5); m++;
    } else for( d=0; d<=dy; d++ ) {
      t=flip?dy-d:d; v[m]=t+ys; u[m]=(int)(xs+s*t+.5); m++;
    }
  }
  /* get points along y-boundary and downsample */
  free(x); free(y); k=m; m=0; double xd, yd;
  x=malloc(sizeof(int)*k); y=malloc(sizeof(int)*k);
  for( j=1; j<k; j++ ) if(u[j]!=u[j-1]) {
    xd=(double)(u[j]<u[j-1]?u[j]:u[j]-1); xd=(xd+.5)/scale-.5;
    if( floor(xd)!=xd || xd<0 || xd>w-1 ) continue;
    yd=(double)(v[j]<v[j-1]?v[j]:v[j-1]); yd=(yd+.5)/scale-.5;
    if(yd<0) yd=0; else if(yd>h) yd=h; yd=ceil(yd);
    x[m]=(int) xd; y[m]=(int) yd; m++;
  }
  /* compute rle encoding given y-boundary points */
  k=m; a=malloc(sizeof(uint)*(k+1));
  for( j=0; j<k; j++ ) a[j]=(uint)(x[j]*(int)(h)+y[j]);
  a[k++]=(uint)(h*w); free(u); free(v); free(x); free(y);
  qsort(a,k,sizeof(uint),uintCompare); uint p=0;
  for( j=0; j<k; j++ ) { uint t=a[j]; a[j]-=p; p=t; }
  b=malloc(sizeof(uint)*k); j=m=0; b[m++]=a[j++];
  while(j<k) if(a[j]>0) b[m++]=a[j++]; else {
    j++; if(j<k) b[m-1]+=a[j++]; }
  rleInit(R,h,w,m,b); free(a); free(b);
}

char* rleToString( const RLE *R ) {
  /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
  siz i, m=R->m, p=0; long x; int more;
  char *s=malloc(sizeof(char)*m*6);
  for( i=0; i<m; i++ ) {
    x=(long) R->cnts[i]; if(i>2) x-=(long) R->cnts[i-2]; more=1;
    while( more ) {
      char c=x & 0x1f; x >>= 5; more=(c & 0x10) ? x!=-1 : x!=0;
      if(more) c |= 0x20; c+=48; s[p++]=c;
    }
  }
  s[p]=0; return s;
}

void rleFrString( RLE *R, char *s, siz h, siz w ) {
  siz m=0, p=0, k; long x; int more; uint *cnts;
  while( s[m] ) m++; cnts=malloc(sizeof(uint)*m); m=0;
  while( s[p] ) {
    x=0; k=0; more=1;
    while( more ) {
      char c=s[p]-48; x |= (c & 0x1f) << 5*k;
      more = c & 0x20; p++; k++;
      if(!more && (c & 0x10)) x |= -1 << 5*k;
    }
    if(m>2) x+=(long) cnts[m-2]; cnts[m++]=(uint) x;
  }
  rleInit(R,h,w,m,cnts); free(cnts);
}
