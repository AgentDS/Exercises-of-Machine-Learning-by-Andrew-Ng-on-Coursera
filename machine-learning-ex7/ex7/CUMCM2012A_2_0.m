% PCA确定提取前2个主成分后构造葡萄的新特征矩阵
X=[2027.96 	553.1058549	0.251	409.708665	2.060 	18.210 	1.830 	33.75282751	1119.852536	0.430120036	23.60445766	22.01903336	9.479519356	3.195 	17.6780 	208.1746032	237.6676964	226.4666667	3.56	5.856666667	38.65603704	25.918	182.93	123.6333333	4.506666667	78.4	0.11	24.06666667	0.78	0.26
2128.82 	626.4780761	0.06215	226.010449	9.930 	4.750 	0.770 	30.90426535	762.5249317	0.464356414	26.87526842	23.36131199	13.80562885	4.889 	27.4550 	205	229.1363657	228.8	3.953333333	5.193333333	44.05271691	25.986	81.61666667	98.3	3.833333333	77.5	0.163333333	26.07	0.646666667	-1.25
8397.28 	585.0456011	0.315	159.6468021	8.080 	2.960 	1.050 	19.30293901	266.6397987	0.408998822	21.6848192	20.37287094	10.79425853	4.764 	164.9927 	256.1904762	273.7582679	257.6333333	3.91	7.16	35.99372124	28.99733333	83.13	105.4	5.596666667	71.83333333	0.17	25.5	1.086666667	-0.616666667
2144.68 	529.823037	0.0967	81.37159488	3.770 	5.230 	0.550 	15.53374439	72.9048704	0.265546732	10.69844628	8.63843655	4.481660528	3.412 	26.9679 	189.7222222	237.7660996	203.3333333	3.29	7.106666667	28.60955966	23.72133333	137.97	174.7	3.263333333	52.96666667	0.174	25.98333333	1.84	-0.366666667
1844.00 	585.6130496	0.0405	122.2929008	9.490 	3.770 	1.440 	31.53569726	143.5133063	0.396096649	17.61781035	14.48577344	10.27473463	0.637 	6.6502 	209.6626984	195.4598914	212.9333333	3.636666667	6.653333333	32.00267231	24.08366667	515.4633333	254.2333333	2.99	65.63333333	0.27	26.33333333	0.88	-0.333333333
3434.17 	536.6428076	0.07485	47.86585357	2.830 	2.210 	0	36.77386353	115.9426525	0.275012207	10.67146721	15.17298506	6.838445477	2.203 	7.7272 	244.3849206	223.8170469	246.1333333	3.286666667	9.313333333	26.42660439	27.37633333	202.2366667	171.9666667	2.636666667	71.93333333	0.193333333	25.16333333	1.806666667	-0.16
2391.16 	487.1719481	0.1305	62.40885356	5.820 	7.740 	0.540 	25.59136163	433.7508416	0.175555132	9.214390599	5.619186275	3.468154441	0.623 	9.8648 	209.8611111	303.9500595	211.3666667	3.18	8.136666667	25.97915393	26.43766667	63.61	168.8333333	4.783333333	71.5	0.141333333	25.60666667	2.05	-0.38
1950.76 	558.545995	0.1805	243.0968666	5.710 	13.550 	2.510 	50.43390128	1305.594766	0.414843621	15.2406428	22.4889913	8.483249026	5.949 	115.5546 	198.8492063	196.9896583	226.4666667	2.92	6.473333333	34.98961401	25.62033333	213.0866667	181.0666667	6.406666667	59.56666667	0.26	26.85	0.803333333	-0.51
2262.72 	700.8279814	0.512	242.4986115	13.230 	4.120 	1.100 	16.86867275	424.108428	0.665753502	30.1140053	24.36227456	20.49005642	4.907 	58.5407 	193.6904762	194.9251356	203.3666667	3.74	5.883333333	34.57578584	23.76133333	186.6166667	138.0666667	5.306666667	77.96666667	0.13	23.81	1.44	-0.376666667
1364.14 	545.3050247	0.119969	45.87147446	2.450 	2.300 	0.240 	10.42658031	459.5685129	0.325517369	9.475944001	16.68770159	4.631478376	12.307 	28.7475 	167.202381	161.4208491	181.2266667	3.65	6.673333333	27.15844814	19.676	255.44	200.8	4.59	71.7	0.2	27.1	2.166666667	-1.12
2355.69 	542.6623486	0.07625	9.42675673	9.290 	8.610 	1.900 	14.26010496	91.46828936	0.279005258	6.074674424	4.543009684	2.516944418	26.851 	25.5751 	209.5634921	237.8913609	210.2	3.533333333	5.496666667	38.2436475	24.527	177.83	118.8	3.41	58.41	0.102	28.02666667	12.15	3.873333333
2556.79 	493.4599127	0.0648	34.0227806	6.080 	5.330 	1.130 	21.07963077	132.2162157	0.197265919	12.05898812	7.168961176	3.896651291	0.696 	2.4802 	247.6587302	262.1546197	261.1	3.433333333	8.536666667	30.58407181	27.61433333	191.9466667	187.7333333	2.396666667	63.3	0.243333333	26.57333333	2.043333333	0.006666667
1416.11 	606.203506	0.01515	67.0096327	4.300 	0.830 	1.150 	28.07621581	99.88098738	0.440591146	14.38525904	9.821984314	7.330450184	10.863 	40.7586 	197.8571429	212.2369872	203.3666667	3.86	4.336666667	23.74991996	23.35266667	159.97	147.9666667	4.673333333	68.1	0.16	27.53	1.043333333	-1.566666667
1237.81 	599.8287783	0.0598	141.9215837	5.730 	4.120 	1.630 	41.57685539	991.0458699	0.359706993	14.65745623	13.94054108	7.808513581	6.313 	134.6375 	191.5079365	255.3353778	193.8666667	3.39	5.4	35.90339162	24.06	209.1066667	136.2666667	4.603333333	66.15333333	0.255	25.41333333	1.193333333	-0.566666667
2177.91 	524.6126706	0.06805	54.45469241	6.230 	3.630 	2.060 	25.74256473	157.9973985	0.218933719	11.9007108	25.41700677	5.511334597	0.211 	9.7179 	179.1071429	208.932908	214.8666667	3.186666667	8.566666667	25.09280466	25.012	159.31	174.4666667	2.903333333	67.69666667	0.213	25.52666667	1.98	-0.01
1553.50 	583.3737451	0.0833	62.31081124	9.030 	7.280 	2.380 	13.64790014	529.9692895	0.236736274	11.21387291	10.08617314	9.156738795	4.556 	8.1900 	204.0079365	189.2752363	205.6333333	3.296666667	4.923333333	41.76066218	22.34566667	119.1733333	109.3333333	3.793333333	71.83333333	0.135	26.11333333	1.333333333	-0.343333333
1713.65 	548.8334348	0.05595	61.1214327	5.880 	5.110 	0.880 	17.17385684	129.5808434	0.358523283	15.33577032	15.73033787	8.700839409	0.711 	43.8121 	212.7380952	271.5038654	238.2	3.43	8.66	27.51201428	26.27633333	446.6366667	264.1	2.796666667	71.53333333	0.33	25.4	1.18	-0.246666667
2398.38 	513.8167484	0.1115	40.08988767	3.60 	5.590 	0.520 	27.07696985	158.8699049	0.225596621	7.380943788	5.388158502	5.244612518	0.416 	6.5161 	226.031746	265.7727031	226.5666667	3.266666667	8.033333333	28.20700369	26.33766667	196.0066667	208.4	2.596666667	63.06666667	0.16	25.52	2.866666667	0.213333333
2463.60 	544.4615091	0.07225	117.3167105	5.560 	4.270 	0.130 	30.40787939	202.9624159	0.379578685	17.42593537	13.70002353	9.454030345	3.821 	31.2649 	205.7936508	220.332557	214.9	3.573333333	6.813333333	31.53722883	23.44133333	173.0933333	168.8333333	6.316666667	67.43333333	0.162333333	27.19333333	0.796666667	-1.513333333
2273.63 	559.3322811	0.02415	25.17378337	3.510 	0.920 	0.440 	12.43860432	89.77036645	0.281917544	12.67698625	8.114932411	8.155068266	1.545 	9.6262 	193.1944444	227.3375389	209.1	3.81	5.166666667	40.47932153	22.93333333	307.1433333	334.3	3.153333333	59.53333333	0.231666667	27.08666667	1.956666667	-0.433333333
6346.83 	563.7941309	0.05035	90.89032213	15.510 	2.930 	2.380 	18.12315975	194.2620763	0.379317681	16.1923429	13.61303727	7.514509794	7.847 	47.2196 	205.7936508	259.110459	216.9	3.563333333	6.783333333	31.99166477	26.94766667	147.66	106.1	4.743333333	60.4	0.108333333	25.18	1.213333333	-0.003333333
2566.61 	488.7123377	0.0742	75.70901356	6.490 	7.730 	0.770 	21.82436421	417.665277	0.283676175	16.44234457	12.15510431	7.846054852	4.289 	13.8003 	224.1468254	226.3990788	234.7	3.646666667	5.966666667	39.36081945	25.67433333	106.6133333	115.8333333	3.32	57.43333333	0.147	25.94	1.52	-0.07
2380.81 	543.5740313	0.0968	174.2620378	4.080 	5.200 	0.390 	16.40605054	427.0280581	0.572487804	29.70416867	24.25693922	24.29491191	9.968 	44.7476 	207.6785714	212.5640163	208.7666667	3.39	6.906666667	30.23256149	23.38333333	278.75	219.0666667	3.836666667	77.46666667	0.233333333	26.64666667	1.383333333	-0.42
1638.83 	525.8201237	0.0328	146.5901639	8.360 	4.600 	1.700 	15.06587834	144.729143	0.282996218	8.750883363	14.41656822	8.205590166	2.935 	14.3803 	201.8253968	244.5123254	203.3333333	3.606666667	7.266666667	27.98330714	25.81533333	517.4544444	237.3666667	2.99	76.73333333	0.246666667	25.96666667	0.896666667	-0.286666667
1409.70 	537.0841804	0.06385	51.27100442	2.870 	2.480 	0.160 	14.27959446	140.945941	0.350894841	11.50244542	9.324297037	5.373138873	2.129 	30.2112 	150.3373016	156.038107	194.6333333	3.383333333	8.533333333	22.80876248	18.51533333	288.69	251.3	4.096666667	58.5	0.220333333	27.09666667	1.52	-0.92
851.17 	587.2932725	0.416	60.16394681	7.150 	1.400 	0.820 	32.02633688	82.35873653	0.317154643	7.347675634	3.777921583	3.383457761	2.086 	13.9166 	173.3531746	197.3769395	195.7333333	3.683333333	4.58	42.73527484	19.75833333	793.4666667	245.5	3.353333333	68.3	0.23	27.99666667	1.093333333	-0.826666667
1116.61 	528.3311404	0.0908	35.83805254	6.230 	1.390 	1.260 	23.0348213	592.198604	0.264875168	8.897079663	10.30959579	4.711185395	1.569 	15.9809 	196.6666667	213.2160525	206.9	3.37	6.973333333	29.6695842	23.329	282.0866667	148.7333333	3.506666667	59.5	0.199666667	28.79	2.333333333	-1.23];

score= ([627	803	804	686	733	722	715	723	815	742	701	539	746	730	587	749	793	599	786	792	771	772	856	780	692	738	730]/10)';
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);
K = 2;
Z = projectData(X_norm, U, K);
plot3(Z(:,1),Z(:,2),score,'o');