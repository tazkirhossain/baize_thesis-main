After preprocessing:
Group	Records
Adult	34,811
Non-adult	1,433,432
Total	1,468,243


The original CSV contained 1,562,978 records. Cleaning excluded:
- 1,047 missing or invalid records.
- 89,385 repeated records from non-conflicting URLs.
- 4,303 records belonging to 148 URLs with conflicting labels.
Exact retained category counts:
Category	  Binary group	        Records
Adult	      Adult	                34,811
Arts	      Non-adult	            237,804
Business	  Non-adult	            235,405
Computers	  Non-adult	            111,101
Games	      Non-adult	            52,546
Health	      Non-adult	            58,023
Home	      Non-adult	            26,886
Kids	      Non-adult	            37,077
News	      Non-adult	            8,373
Recreation	  Non-adult	            102,423
Reference	  Non-adult	            53,654
Science	      Non-adult	            97,763
Shopping	  Non-adult	            93,233
Society	      Non-adult	            226,372
Sports	      Non-adult	            92,772
Total		                        1,468,243


Distribution across clients:
Client	Adult	Non-adult	Total	Train	 Validation	Test
0	    6,963	286,687	    293,650	205,555	 44,048	    44,047
1	    6,962	286,687	    293,649	205,555	 44,047	    44,047
2	    6,962	286,686	    293,648	205,554	 44,047	    44,047
3	    6,962	286,686	    293,648	205,554	 44,047	    44,047
4	    6,962	286,686	    293,648	205,554	 44,047	    44,047