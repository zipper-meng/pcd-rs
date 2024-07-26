//! Types for reading PCD data.
//!
//! [Reader](crate::reader::Reader) lets you load points sequentially with
//! [Iterator](std::iter::Iterator) interface. The points are stored in
//! types implementing [PcdDeserialize](crate::record::PcdDeserialize) trait.
//! See [record](crate::record) moduel doc to implement your own point type.
#![cfg_attr(
    feature = "derive",
    doc = r##"
```rust
use anyhow::Result;
use pcd_rs::{PcdDeserialize, Reader};
use std::path::Path;

#[derive(PcdDeserialize)]
pub struct Point {
    x: f32,
    y: f32,
    z: f32,
    rgb: f32,
}

fn main() -> Result<()> {
    let reader = Reader::open("test_files/ascii.pcd")?;
    let points: Result<Vec<Point>> = reader.collect();
    assert_eq!(points?.len(), 213);
    Ok(())
}
```
"##
)]

use crate::{
    error::Error,
    metas::{DataKind, FieldDef, PcdMeta},
    record::{DynRecord, PcdDeserialize},
};
use anyhow::Result;
use std::{
    fs::File,
    io::{prelude::*, BufReader, Cursor},
    marker::PhantomData,
    path::Path,
};

/// The `DynReader` struct loads points with schema determined in runtime.
pub type DynReader<R> = Reader<DynRecord, R>;

/// The `Reader<T, R>` struct loads points into type `T` from reader `R`.
pub struct Reader<T, R>
where
    R: Read,
{
    meta: PcdMeta,
    record_count: usize,
    finished: bool,
    reader: R,
    #[cfg(feature = "binary_compressed")]
    decompressed_chunk: std::collections::VecDeque<T>,
    _phantom: PhantomData<T>,
}

impl<'a, Record> Reader<Record, BufReader<Cursor<&'a [u8]>>>
where
    Record: PcdDeserialize,
{
    pub fn from_bytes(buf: &'a [u8]) -> Result<Self> {
        let reader = BufReader::new(Cursor::new(buf));
        Self::from_reader(reader)
    }
}

impl<Record, R> Reader<Record, R>
where
    Record: PcdDeserialize,
    R: BufRead,
{
    pub fn from_reader(mut reader: R) -> Result<Self> {
        let mut line_count = 0;
        let meta = crate::utils::load_meta(&mut reader, &mut line_count)?;

        // Checks whether the record schema matches the file meta
        if !Record::is_dynamic() {
            let record_spec = Record::read_spec();

            let mismatch_error =
                Error::new_schema_mismatch_error(record_spec.as_slice(), &meta.field_defs.fields);

            if record_spec.len() != meta.field_defs.len() {
                return Err(mismatch_error.into());
            }

            for (record_field, meta_field) in record_spec.into_iter().zip(meta.field_defs.iter()) {
                let (name_opt, record_kind, record_count_opt) = record_field;
                let FieldDef {
                    name: meta_name,
                    kind: meta_kind,
                    count: meta_count,
                } = meta_field;

                if record_kind != *meta_kind {
                    return Err(mismatch_error.into());
                }

                if let Some(name) = &name_opt {
                    if name != meta_name {
                        return Err(mismatch_error.into());
                    }
                }

                if let Some(record_count) = record_count_opt {
                    if record_count != *meta_count as usize {
                        return Err(mismatch_error.into());
                    }
                }
            }
        }

        let pcd_reader = Reader {
            meta,
            reader,
            record_count: 0,
            finished: false,
            #[cfg(feature = "binary_compressed")]
            decompressed_chunk: std::collections::VecDeque::new(),
            _phantom: PhantomData,
        };

        Ok(pcd_reader)
    }
}

impl<Record> Reader<Record, BufReader<File>>
where
    Record: PcdDeserialize,
{
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = BufReader::new(File::open(path.as_ref())?);
        Self::from_reader(file)
    }
}

impl<R, Record> Reader<Record, R>
where
    R: BufRead,
{
    /// Get meta data.
    pub fn meta(&self) -> &PcdMeta {
        &self.meta
    }
}

impl<R, Record> Iterator for Reader<Record, R>
where
    R: BufRead,
    Record: PcdDeserialize,
{
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let record_result = match self.meta.data {
            DataKind::Ascii => Record::read_line(&mut self.reader, &self.meta.field_defs),
            DataKind::Binary => Record::read_chunk(&mut self.reader, &self.meta.field_defs),
            #[cfg(feature = "binary_compressed")]
            DataKind::BinaryCompressed => {
                let mut read_result: Option<Result<Record>> = None;
                if self.decompressed_chunk.is_empty() {
                    if let Err(e) = Record::read_compressed_chunk(
                        &mut self.reader,
                        &self.meta,
                        &mut self.decompressed_chunk,
                    ) {
                        read_result = Some(Err(e));
                    }
                }
                match read_result {
                    Some(err) => err,
                    None => match Record::read_decompressed_chunk(&mut self.decompressed_chunk) {
                        Some(record) => Ok(record),
                        None => {
                            self.finished = true;
                            return None;
                        }
                    },
                }
            }
        };

        match record_result {
            Ok(_) => {
                self.record_count += 1;
                if self.record_count == self.meta.num_points as usize {
                    self.finished = true;
                }
            }
            Err(_) => {
                self.finished = true;
            }
        }

        Some(record_result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.meta.num_points as usize;
        (size, Some(size))
    }
}
