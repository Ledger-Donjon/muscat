//! Load and iterate over traces and data from [quicklog](https://github.com/Ledger-Donjon/quicklog) log files.

use ndarray::Array1;
use npyz::{Deserialize, NpyFile};
use serde_json::map::IntoIter;
use std::{
    fs::File,
    io::{BufRead, BufReader, Error, Lines, Seek, SeekFrom},
    marker::PhantomData,
    path::Path,
    time::Instant,
};

use crate::{trace::Trace, util::read_array_1_from_npy_file};

/// Returns traces database directory from `TRACESDIR` environment variable, or `None` if it is not
/// defined.
fn get_traces_dir() -> Option<String> {
    match std::env::var("TRACESDIR") {
        Ok(dir) => Some(dir),
        Err(_) => None,
    }
}

/// Returns the path of a trace or trace batch in the database from an id.
pub fn path_from_string_id(id: &[u8]) -> String {
    let traces_dir = get_traces_dir().expect("TRACESDIR is not defined");
    Path::new(&traces_dir)
        .join(format!("{:02x}", id[0]))
        .join(format!("{:02x}", id[1]))
        .join(format!("{:02x}", id[2]))
        .join(format!("{:02x}", id[3]))
        .join(hex::encode(&id[4..]) + ".npy")
        .to_str()
        .unwrap()
        .to_string()
}

pub fn guess_leakages_size<T: Deserialize>(path: &str) -> usize {
    let record = FileRecordIterator::<T>::new(path)
        .expect("Failed to open log")
        .next()
        .expect("Failed to fetch first log entry")
        .expect("Failed to parse first log entry");
    let leakage = record.load_trace().expect("Failed to load first log trace");
    leakage.len()
}

/// Parsing records log can produce to types of errors, wrapped into this single error type.
#[derive(Debug)]
pub enum LogError {
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    NoRecords,
}

impl From<std::io::Error> for LogError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError(error)
    }
}

impl From<serde_json::Error> for LogError {
    fn from(error: serde_json::Error) -> Self {
        Self::JsonError(error)
    }
}

/// Opens a log file and allows iterating over the records.
///
/// `T` specifies the type of the elements in the leakages.
pub struct Log<T> {
    records: Vec<Record<T>>,
    leakage_size: usize,
    phantom: PhantomData<T>,
}

impl<T: Deserialize> Log<T> {
    /// Opens a log file and parses all the entries. Gets the length of leakages
    /// by reading the first trace.
    pub fn new(path: &str) -> Result<Self, LogError> {
        let mut records = Vec::new();
        for entry in FileRecordIterator::new(path)? {
            match entry {
                Ok(record) => records.push(record),
                Err(e) => return Err(e),
            }
        }
        if let Some(record) = records.first() {
            let leakage_size = record.load_trace()?.len();
            Ok(Self {
                records,
                leakage_size,
                phantom: PhantomData,
            })
        } else {
            Err(LogError::NoRecords)
        }
    }

    /// Returns the number of records in the log
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns the number of samples in each leakage
    pub fn leakage_size(&self) -> usize {
        self.leakage_size
    }
}

impl<T> IntoIterator for Log<T> {
    type Item = Record<T>;

    type IntoIter = std::vec::IntoIter<Record<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.into_iter()
    }
}

/// A record element from a quicklog log file
///
/// `T` specifies the type of the elements in the leakages.
pub struct Record<T> {
    pub data: serde_json::Value,
    phantom: PhantomData<T>,
}

impl<T: Deserialize> Record<T> {
    /// Returns a bytes identifier from the record data
    ///
    /// # Arguments
    ///
    /// * `key` - String key, such as "bid" or "tid".
    pub fn get_id(&self, key: &str) -> Option<Vec<u8>> {
        self.data
            .get(key)
            .map(|id| hex::decode(id.as_str().unwrap()).expect("Invalid id hex string"))
    }

    /// Returns the trace id, if defined.
    pub fn tid(&self) -> Option<Vec<u8>> {
        self.get_id("tid")
    }

    /// Returns the trace batch id, if defined.
    pub fn bid(&self) -> Option<Vec<u8>> {
        self.get_id("bid")
    }

    /// Returns the offset of the trace data in its batch file.
    ///
    /// Panics if the trace is not stored in a batch
    pub fn toff(&self) -> u64 {
        self.data
            .get("toff")
            .expect("Trace offset missing from record")
            .as_u64()
            .expect("Invalid trace offset format")
    }

    /// Returns bytes encoded in hex in a given field of the record. Panics if the data does not
    /// exist, is not a string or is not hex encoded.
    ///
    /// # Arguments
    ///
    /// * `key` - JSON field key
    pub fn bytes(&self, key: &str) -> Vec<u8> {
        hex::decode(self.data[key].as_str().unwrap()).expect("Failed to parse bytes in record")
    }

    pub fn load_trace(&self) -> Result<Array1<T>, LogError> {
        if let Some(bid) = self.bid() {
            // Trace is stored in a batch
            let toff = self.toff();
            let path = path_from_string_id(&bid);
            let mut f = File::open(path)?;
            f.seek(SeekFrom::Start(toff)).unwrap();
            let buf = BufReader::new(f);
            let npy = NpyFile::new(buf).unwrap();
            Ok(read_array_1_from_npy_file(npy))
        } else if let Some(tid) = self.tid() {
            // Trace is stored in a single file
            todo!()
        } else {
            // Cannot find a trace in this record
            todo!()
        }
    }
}

pub struct FileRecordIterator<T> {
    lines: Lines<BufReader<File>>,
    phantom: PhantomData<T>,
}

impl<T> FileRecordIterator<T> {
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self {
            lines: reader.lines(),
            phantom: PhantomData,
        })
    }
}

impl<T> Iterator for FileRecordIterator<T> {
    type Item = Result<Record<T>, LogError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(line) = self.lines.next() {
            match line {
                Ok(line) => {
                    let value: Result<serde_json::Value, _> = serde_json::from_str(line.as_str());
                    match value {
                        Ok(value) => Some(Ok(Record {
                            data: value,
                            phantom: PhantomData,
                        })),
                        Err(err) => Some(Err(LogError::JsonError(err))),
                    }
                }
                Err(err) => Some(Err(LogError::IoError(err))),
            }
        } else {
            None
        }
    }
}

pub struct CachedLoader {
    current_path: Option<String>,
    current_data: Vec<u8>,
}

impl CachedLoader {
    pub fn new() -> Self {
        Self {
            current_path: None,
            current_data: Vec::new(),
        }
    }

    pub fn load_trace<T: Deserialize>(
        &mut self,
        record: &Record<T>,
    ) -> Result<Array1<T>, LogError> {
        if let Some(bid) = record.bid() {
            let path = path_from_string_id(&bid);
            if self.current_path != Some(path.clone()) {
                self.current_data = std::fs::read(&path).unwrap();
                self.current_path = Some(path)
            }
            let toff = record.toff();
            let start = Instant::now();
            let chunk = &self.current_data.as_slice()[toff as usize..];
            let npy = NpyFile::new(chunk).unwrap();
            Ok(read_array_1_from_npy_file(npy))
        } else {
            record.load_trace()
        }
    }
}

/// Holds a trace batch file content and an offset list in the file, plus the
/// data associated to each trace.
///
/// This can be created by [BatchIterator].
pub struct Batch<T, U> {
    file: Vec<u8>,
    toffs_and_values: Vec<(usize, U)>,
    phantom: PhantomData<T>,
}

impl<T, U> Batch<T, U> {
    fn new() -> Self {
        Self {
            file: Vec::new(),
            toffs_and_values: Vec::new(),
            phantom: PhantomData,
        }
    }
}

impl<T: Deserialize, U> IntoIterator for Batch<T, U> {
    type Item = Trace<T, U>;

    type IntoIter = BatchTraceIterator<T, U>;

    fn into_iter(self) -> Self::IntoIter {
        BatchTraceIterator {
            bytes: self.file,
            iter: self.toffs_and_values.into_iter(),
            phantom: PhantomData,
        }
    }
}

/// Iterates log files to generate [`Batch`] that groups traces by batch files.
///
/// `T` is the type of the elements in the leakage arrays.
/// `U` is the type of data associated to each leakage.
pub struct BatchIterator<T, U, I, F>
where
    I: Iterator<Item = Record<T>>,
    F: Fn(Record<T>) -> U,
{
    /// A closure applied to records to select trace associated data
    f: F,
    /// First record of the next batch
    first: Option<Record<T>>,
    /// Internal state iterator
    records: I,
    phantom: PhantomData<T>,
}

impl<T, U, I, F> BatchIterator<T, U, I, F>
where
    I: Iterator<Item = Record<T>>,
    F: Fn(Record<T>) -> U,
{
    pub fn new(mut iter: I, f: F) -> Self {
        Self {
            f,
            first: iter.next(),
            records: iter,
            phantom: PhantomData,
        }
    }
}

impl<T, U, I, F> Iterator for BatchIterator<T, U, I, F>
where
    T: Deserialize,
    I: Iterator<Item = Record<T>>,
    F: Fn(Record<T>) -> U,
{
    type Item = Batch<T, U>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(first) = self.first.take() {
            let fid = first.bid().unwrap();
            let path = path_from_string_id(&fid);
            let mut batch = Batch {
                file: std::fs::read(path).unwrap(),
                toffs_and_values: vec![(first.toff() as usize, (self.f)(first))],
                phantom: PhantomData,
            };
            loop {
                if let Some(next) = self.records.next() {
                    if next.bid().unwrap() != fid {
                        self.first = Some(next);
                        return Some(batch);
                    } else {
                        batch
                            .toffs_and_values
                            .push((next.toff() as usize, (self.f)(next)));
                    }
                } else {
                    return Some(batch);
                }
            }
        } else {
            None
        }
    }
}

pub trait BatchIter<T, U, I, F>
where
    I: Iterator<Item = Record<T>>,
    F: Fn(Record<T>) -> U,
{
    fn batches(self, f: F) -> BatchIterator<T, U, I, F>;
}

/// Implement [BatchIter] for all [Record] iterators.
impl<T, U, I: Iterator<Item = Record<T>>, F> BatchIter<T, U, Self, F> for I
where
    F: Fn(Record<T>) -> U,
{
    fn batches(self, f: F) -> BatchIterator<T, U, Self, F> {
        BatchIterator::new(self, f)
    }
}

pub struct BatchTraceIterator<T: Deserialize, U> {
    bytes: Vec<u8>,
    iter: std::vec::IntoIter<(usize, U)>,
    phantom: PhantomData<T>,
}

impl<T: Deserialize, U> Iterator for BatchTraceIterator<T, U> {
    type Item = Trace<T, U>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(toff, value)| Trace::new(array_from_bytes(&self.bytes, toff), value))
    }
}

pub fn array_from_bytes<T: Deserialize>(bytes: &[u8], toff: usize) -> Array1<T> {
    let chunk = &bytes[toff as usize..];
    let npy = NpyFile::new(chunk).unwrap();
    read_array_1_from_npy_file(npy)
}
