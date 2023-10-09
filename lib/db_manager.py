#!/usr/env/python3
import sqlite3 as sql
import os,sys
import argparse

#tables:
#   files:
#        name ,path,key (NULL), scriptname,scriptpath,created,counter,data_used(NULL),comment (NULL)
#   plots:
#       name,path,scriptname,scriptpath,created,counter,datafile(NULL), comment(NULL)
#

DB_FILE = os.getenv("THESIS_DATABASE_FILE")
if not os.path.isfile(DB_FILE):
    print("Error: Database file not found")
else:
    con = sql.connect(DB_FILE)
    cursor = con.cursor()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Extracts information from the plots/files database")
    parser.add_argument("-p", "--plot", help="Get information about plot", type=str)
    parser.add_argument("-f", "--files", help="Get information about file", type=str)
    parser.add_argument("-t", "--path", help="Get information about a path", type=str)
    parser.add_argument("-n", "--nlines", help="Max number of results", type=str)

    args = vars(parser.parse_args())
    if(args["plot"]):
        query = "SELECT * FROM plots WHERE name LIKE :plot ORDER BY created DESC"
        if(args["nlines"]):
            query += " LIMIT :nlines"
        res = cursor.execute(query, dict(plot=args['plot'], nlines=args['nlines']))
        for line in res:
            print(line)
    if(args["path"]):
        query = "SELECT * FROM files WHERE scriptpath LIKE :path OR path like :path ORDER BY created DESC"
        if(args["nlines"]):
            query += " LIMIT :nlines"
        res = cursor.execute(query, dict(path=args['path'], nlines=args['nlines']))
        print("Files created by scripts in path:")
        for line in res:
            print(line)
            
        query = "SELECT * FROM runs WHERE scriptpath LIKE :path ORDER BY time DESC"
        if(args["nlines"]):
            query += " LIMIT :nlines"
        res = cursor.execute(query, dict(path=args['path'], nlines=args['nlines']))
        print("Runs of scripts in path:")
        for line in res:
            print(line)
    if(args["files"]):
        query = "SELECT * FROM files WHERE name LIKE :file ORDER BY created DESC"
        if(args["nlines"]):
            query += " LIMIT :nlines"
        print(query)    
        res = cursor.execute(query, dict(file=args['files'], nlines=args['nlines']))
        print(res.fetchall())
        for line in res:
            print(line)
#rename_query = "UPDATE runs SET scriptpath=:new WHERE scriptpath LIKE :old"
#res = cursor.execute(rename_query, dict(new="/net/home/lxtsfs1/tpe/geuskens/Analysis/train/particlenet", old="%/train"))
#con.commit()
#print(res)
