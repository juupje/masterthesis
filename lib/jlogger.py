"""
DEPRICATED! THIS IS ONLY HERE FOR BACKWARD COMPATIBILITY FOR OLD RUNS
FOR NEW RUNS, USE `steganologger` https://github.com/juupje/steganologger
"""
import sqlite3 as sq
import sys, os
from datetime import datetime
import h5py

DATA_DIR = os.getenv("DATA_DIR")
DB_FILE = os.getenv("THESIS_DATABASE_FILE")

class JLogger:
    def __init__(self, storage_id:str=None, database_file:str=DB_FILE, jobid:int=None, output_dir:str=None, comment:str=None):
        if(not(storage_id)):
            storage_id = sys.argv[0]
        if(storage_id.endswith(".py")):
            storage_id = storage_id[:-3]
        elif(storage_id.endswith(".ipynb")):
            storage_id = storage_id[:-6]
        self.storage_id = storage_id

        if not os.path.isfile(database_file):
            self.con = sq.connect(database_file)
            self.cursor = self.con.cursor()
            self.cursor.execute("""
                        CREATE TABLE runs(
                                    scriptname NOT NULL,
                                    scriptpath NOT NULL,
                                    time NOT NULL,
                                    job_id,
                                    output_dir,
                                    comment)
                """)
            self.cursor.execute("""
                        CREATE TABLE files(name NOT NULL,
                                    path NOT NULL,
                                    key,
                                    scriptname NOT NULL,
                                    scriptpath NOT NULL,
                                    created NOT NULL,
                                    counter NOT NULL,
                                    data_used,
                                    comment)
                """)
            self.cursor.execute("""
                        CREATE TABLE plots(name NOT NULL,
                                    path NOT NULL,
                                    scriptname NOT NULL,
                                    scriptpath NOT NULL,
                                    created NOT NULL,
                                    counter NOT NULL,
                                    datafile, comment)
                """)
            self.con.commit()
        else:
            self.con = sq.connect(database_file)
            self.cursor = self.con.cursor()
        self.jobid = jobid
        self.timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.cursor.execute("INSERT INTO runs VALUES(:sname, :spath, :time, :job, :outdir, :comment)",
                dict(sname=os.path.basename(sys.argv[0]),
                    spath=os.path.dirname(os.path.abspath(sys.argv[0])),
                    time=self.timestamp,
                    job=jobid,
                    outdir=output_dir,
                    comment=comment
                    ))
        self.con.commit()
        self.file = None
        self.file_path = None
        self.plot_counter = 0
        self.file_counter = 0
        print(f"Logger initialized with database file {database_file} and script id {self.storage_id}")

    def exists_data(self, key:str, group_name:str=None):
        if self.file is None:
            if(not os.path.isfile(DATA_DIR+"/processed/"+self.storage_id+".h5")):
                return False
            self._open_file()
        group = None
        if(group_name):
            if(group_name not in self.file.keys()):
                return False
            else:
                group = self.file[group_name]
        else:
            group = self.file
        if(group is None):
            return False
        return key in group.keys()

    def _open_file(self,mode="a"):
        self.file_path = DATA_DIR+"/processed/"+self.storage_id+".h5"
        self.file = h5py.File(self.file_path, mode)
        return self.file

    def store_log_data(self, data:dict, group_name:str=None, comment:str=None, data_used:str=None):
        """
        Stores the data in an hdf5 file and adds an entry to the database for the data being saved
        
        Parameters
        ----------
        data : dict or str
            strings as keys and numpy objects as values.
            The dictionary is saved as an hdf5 file named after the script which is being executed
        comment : str, optional
            A comment which is added to the database entry
        data_used : str, optional
            The source data which was used to create the data which was stored
        """

        #first we're gonna save the data in an hdf5 file
        if not self.file:
            self._open_file()

        group = None
        if(group_name):
            if(group_name not in self.file.keys()):
                group = self.file.create_group(group_name)
            else:
                group = self.file[group_name]
        else:
            group = self.file
        for key in data:
            if(key in group.keys()):
                del group[key]
            group.create_dataset(key, data=data[key])
        self.log_data(self.file_path, f"{group_name}/{key}" if group_name else key, comment, data_used)

    def log_data(self, fname, key:str=None, comment:str=None, data_used:str=None):
        """
        Adds a log the the database that data was stored to this file
        """
        self.file_counter += 1
        try:
            data = {
                "name": os.path.basename(fname),
                "path": os.path.dirname(os.path.abspath(fname)),
                "key": key,
                "sname": os.path.basename(sys.argv[0]),
                "spath": os.path.dirname(os.path.abspath(sys.argv[0])),
                "created": self.timestamp,
                "data_used": data_used,
                "counter": self.file_counter,
                "comment": comment
            }
            self.cursor.execute("INSERT INTO files VALUES(:name, :path, :key, :sname, :spath, :created, :counter, :data_used, :comment)", data)
            self.con.commit()
        except Exception as error:
            print("Something went wrong! Log might not be saved")
            print(error)

    def retrieve_logged_data(self, key:str, group_name:str=None):
        """
        This return value of this method should not be modified!
        """
        if not self.file:
            self._open_file()

        if(not self.exists_data(key, group_name=group_name)):
            raise ValueError(f"No dataset {group_name}/{key} found! Available groups: \n\t" + ", ".join(list(self.file[group_name].keys() if self.exists_data(group_name) else self.file.keys())))
        return self.file[group_name][key] if group_name is not None else self.file[key]

    def log_figure(self, figure, filepath:str, data_file:str = "auto", comment:str=None, **kwargs):
        if(figure is not None):
            figure.savefig(filepath, **kwargs)
        self.plot_counter += 1
        if(data_file == "auto"):
            data_file = self.file_path if self.file is not None else None
        try:
            data = {
                "name": os.path.basename(filepath),
                "path": os.path.dirname(os.path.abspath(filepath)),
                "sname": os.path.basename(sys.argv[0]),
                "spath": os.path.dirname(os.path.abspath(sys.argv[0])),
                "created": self.timestamp,
                "counter": self.plot_counter,
                "datafile": data_file,
                "comment": comment
            }
            self.cursor.execute("INSERT INTO plots VALUES(:name, :path, :sname, :spath, :created, :counter, :datafile, :comment)", data)
            self.con.commit()
        except Exception as error:
            print("Something went wrong! Log might not be saved")
            print(error)
        