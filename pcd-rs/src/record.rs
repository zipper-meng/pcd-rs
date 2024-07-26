#![doc = r##"
Defines serializing and deserializing traits and common record types.

Any object scanned by readers or written by writers must implement
[PcdDeserialize](crate::record::PcdDeserialize) or [PcdSerialize](crate::record::PcdSerialize)
respectively.

These traits are not intended to implemented manually.
Please use derive macro instead. For example,

"##]
#![cfg_attr(
    feature = "derive",
    doc = r##"
```rust
use pcd_rs::{PcdDeserialize, PcdSerialize};

#[derive(PcdDeserialize, PcdSerialize)]
pub struct TimestampedPoint {
    x: f32,
    y: f32,
    z: f32,
    timestamp: u32,
}
```
"##
)]
#![doc = r##"
The derive macro accepts normal structs and tuple structs, but does not accept unit structs.

[PcdDeserialize](crate::record::PcdDeserialize) allows fields with either primitive type,
array of primitive type or [Vec](<std::vec::Vec>) of primitive type.

[PcdSerialize](crate::record::PcdSerialize) allows fields with either primitive type or
array of primitive type. The [Vec](<std::vec::Vec>) is ruled out since the length
is not determined in compile-time.

Make sure struct field names match the `FIELDS` header in PCD data.
Otherwise it panics at runtime. You can specify the exact name in header or bypass name check
with attributes. The name check are automatically disabled for tuple structs.
"##]
#![cfg_attr(
    feature = "derive",
    doc = r##"
```rust
use pcd_rs::PcdDeserialize;

#[derive(PcdDeserialize)]
pub struct TimestampedPoint {
    x: f32,
    y: f32,
    z: f32,
    #[pcd(rename = "true_name")]
    rust_name: u32,
    #[pcd(ignore)]
    whatever_name: u32,
}
```
"##
)]
#[cfg(feature = "binary_compressed")]
use crate::metas::PcdMeta;
use crate::{
    error::Error,
    metas::{FieldDef, Schema, ValueKind},
    traits::Value,
};
use anyhow::{bail, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use num_traits::NumCast;
use std::io::prelude::*;

/// [PcdDeserialize](crate::record::PcdDeserialize) is analogous to a _point_ returned from a reader.
///
/// The trait is not intended to be implemented from scratch. You must
/// derive the implementation with `#[derive(PcdDeserialize)]`.
///
/// When the PCD data is in Ascii mode, the record is represented by a line of literals.
/// Otherwise if the data is in binary mode, the record is represented by a fixed size chunk.
pub trait PcdDeserialize: Sized {
    fn is_dynamic() -> bool;
    fn read_spec() -> Vec<(Option<String>, ValueKind, Option<usize>)>;
    fn read_chunk<R: BufRead>(reader: &mut R, field_defs: &Schema) -> Result<Self>;
    fn read_line<R: BufRead>(reader: &mut R, field_defs: &Schema) -> Result<Self>;

    #[cfg(feature = "binary_compressed")]
    fn read_compressed_chunk<R: BufRead>(reader: &mut R, pcd_meta: &PcdMeta) -> Result<Vec<Field>>;
    #[cfg(feature = "binary_compressed")]
    fn read_decompressed_chunk(
        fields_data: &[Field],
        index: usize,
        field_defs: &Schema,
    ) -> Option<Self>;
}

/// [PcdSerialize](crate::record::PcdSerialize) is analogous to a _point_ written by a writer.
///
/// The trait is not intended to be implemented from scratch. You must
/// derive the implementation with `#[derive(PcdSerialize)]`.
///
/// When the PCD data is in Ascii mode, the record is represented by a line of literals.
/// Otherwise if the data is in binary mode, the record is represented by a fixed size chunk.
pub trait PcdSerialize: Sized {
    fn is_dynamic() -> bool;
    fn write_spec() -> Schema;
    fn write_chunk<R: Write + Seek>(&self, writer: &mut R, spec: &Schema) -> Result<()>;
    fn write_line<R: Write + Seek>(&self, writer: &mut R, spec: &Schema) -> Result<()>;

    #[cfg(feature = "binary_compressed")]
    fn write_one_field<R: Write + Seek>(
        &self,
        writer: &mut R,
        spec: &Schema,
        field_index: usize,
    ) -> Result<()>;
}

// Runtime record types

/// An enum representation of untyped data fields.
#[derive(Debug, Clone, PartialEq)]
pub enum Field {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Field {
    pub fn kind(&self) -> ValueKind {
        use Field as F;
        use ValueKind as K;

        match self {
            F::I8(_) => K::I8,
            F::I16(_) => K::I16,
            F::I32(_) => K::I32,
            F::U8(_) => K::U8,
            F::U16(_) => K::U16,
            F::U32(_) => K::U32,
            F::F32(_) => K::F32,
            F::F64(_) => K::F64,
        }
    }

    pub fn count(&self) -> usize {
        use Field as F;

        match self {
            F::I8(values) => values.len(),
            F::I16(values) => values.len(),
            F::I32(values) => values.len(),
            F::U8(values) => values.len(),
            F::U16(values) => values.len(),
            F::U32(values) => values.len(),
            F::F32(values) => values.len(),
            F::F64(values) => values.len(),
        }
    }

    pub fn to_value<T>(&self) -> Option<T>
    where
        T: Value + NumCast,
    {
        use Field as F;

        if T::KIND != self.kind() {
            return None;
        }

        Some(match self {
            F::I8(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
            F::I16(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
            F::I32(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
            F::U8(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
            F::U16(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
            F::U32(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
            F::F32(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
            F::F64(v) => match &**v {
                &[t] => T::from(t)?,
                _ => return None,
            },
        })
    }
}

/// Represents an untyped _point_ in PCD data.
#[derive(Debug, Clone, PartialEq)]
pub struct DynRecord(pub Vec<Field>);

impl DynRecord {
    pub fn is_schema_consistent(&self, schema: &Schema) -> bool {
        if self.0.len() != schema.len() {
            return false;
        }

        self.0
            .iter()
            .zip(schema.iter())
            .all(|(field, schema_field)| {
                use Field as F;
                use ValueKind as K;

                if field.count() != schema_field.count as usize {
                    return false;
                }

                matches!(
                    (field, schema_field.kind),
                    (F::I8(_), K::I8)
                        | (F::I16(_), K::I16)
                        | (F::I32(_), K::I32)
                        | (F::U8(_), K::U8)
                        | (F::U16(_), K::U16)
                        | (F::U32(_), K::U32)
                        | (F::F32(_), K::F32)
                        | (F::F64(_), K::F64)
                )
            })
    }

    pub fn to_xyz<T>(&self) -> Option<[T; 3]>
    where
        T: Value + NumCast,
    {
        use Field as F;

        if self.0.first()?.kind() != T::KIND {
            return None;
        }

        Some(match &*self.0 {
            [F::I8(xv), F::I8(yv), F::I8(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            [F::I16(xv), F::I16(yv), F::I16(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            [F::I32(xv), F::I32(yv), F::I32(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            [F::U8(xv), F::U8(yv), F::U8(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            [F::U16(xv), F::U16(yv), F::U16(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            [F::U32(xv), F::U32(yv), F::U32(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            [F::F32(xv), F::F32(yv), F::F32(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            [F::F64(xv), F::F64(yv), F::F64(zv), ..] => match (&**xv, &**yv, &**zv) {
                (&[x], &[y], &[z]) => [T::from(x)?, T::from(y)?, T::from(z)?],
                _ => return None,
            },
            _ => return None,
        })
    }
}

impl PcdSerialize for DynRecord {
    fn is_dynamic() -> bool {
        true
    }

    fn write_spec() -> Schema {
        unreachable!();
    }

    fn write_chunk<Writer>(&self, writer: &mut Writer, spec: &Schema) -> Result<()>
    where
        Writer: Write + Seek,
    {
        if !self.is_schema_consistent(spec) {
            bail!("The content of record does not match the writer schema.");
        }

        for field in self.0.iter() {
            use Field as F;

            match field {
                F::I8(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_i8(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
                F::I16(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_i16::<LittleEndian>(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
                F::I32(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_i32::<LittleEndian>(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
                F::U8(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_u8(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
                F::U16(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_u16::<LittleEndian>(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
                F::U32(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_u32::<LittleEndian>(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
                F::F32(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_f32::<LittleEndian>(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
                F::F64(values) => {
                    values
                        .iter()
                        .map(|val| Ok(writer.write_f64::<LittleEndian>(*val)?))
                        .collect::<Result<Vec<_>>>()?;
                }
            }
        }

        Ok(())
    }

    fn write_line<Writer>(&self, writer: &mut Writer, spec: &Schema) -> Result<()>
    where
        Writer: Write + Seek,
    {
        if !self.is_schema_consistent(spec) {
            bail!("The content of record does not match the writer schema.");
        }

        let mut tokens = vec![];

        for field in self.0.iter() {
            use Field as F;

            match field {
                F::I8(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
                F::I16(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
                F::I32(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
                F::U8(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
                F::U16(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
                F::U32(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
                F::F32(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
                F::F64(values) => {
                    let iter = values.iter().map(|val| val.to_string());
                    tokens.extend(iter);
                }
            }
        }

        writeln!(writer, "{}", tokens.join(" "))?;

        Ok(())
    }

    #[cfg(feature = "binary_compressed")]
    fn write_one_field<R: Write + Seek>(
        &self,
        writer: &mut R,
        spec: &Schema,
        field_index: usize,
    ) -> Result<()> {
        if !self.is_schema_consistent(spec) {
            bail!("The content of record does not match the writer schema.");
        }

        if field_index > self.0.len() {
            bail!("Field index {field_index} out of range {}.", self.0.len());
        }

        use Field as F;

        match &self.0[field_index] {
            F::I8(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_i8(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
            F::I16(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_i16::<LittleEndian>(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
            F::I32(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_i32::<LittleEndian>(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
            F::U8(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_u8(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
            F::U16(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_u16::<LittleEndian>(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
            F::U32(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_u32::<LittleEndian>(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
            F::F32(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_f32::<LittleEndian>(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
            F::F64(values) => {
                values
                    .iter()
                    .map(|val| Ok(writer.write_f64::<LittleEndian>(*val)?))
                    .collect::<Result<Vec<_>>>()?;
            }
        }

        Ok(())
    }
}

impl PcdDeserialize for DynRecord {
    fn is_dynamic() -> bool {
        true
    }

    fn read_spec() -> Vec<(Option<String>, ValueKind, Option<usize>)> {
        unreachable!();
    }

    fn read_chunk<R: BufRead>(reader: &mut R, field_defs: &Schema) -> Result<Self> {
        use Field as F;
        use ValueKind as K;

        let fields = field_defs
            .iter()
            .map(|def| {
                let FieldDef { kind, count, .. } = *def;

                let counter = 0..count;

                let field = match kind {
                    K::I8 => {
                        let values = counter
                            .map(|_| Ok(reader.read_i8()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::I8(values)
                    }
                    K::I16 => {
                        let values = counter
                            .map(|_| Ok(reader.read_i16::<LittleEndian>()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::I16(values)
                    }
                    K::I32 => {
                        let values = counter
                            .map(|_| Ok(reader.read_i32::<LittleEndian>()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::I32(values)
                    }
                    K::U8 => {
                        let values = counter
                            .map(|_| Ok(reader.read_u8()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::U8(values)
                    }
                    K::U16 => {
                        let values = counter
                            .map(|_| Ok(reader.read_u16::<LittleEndian>()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::U16(values)
                    }
                    K::U32 => {
                        let values = counter
                            .map(|_| Ok(reader.read_u32::<LittleEndian>()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::U32(values)
                    }
                    K::F32 => {
                        let values = counter
                            .map(|_| Ok(reader.read_f32::<LittleEndian>()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::F32(values)
                    }
                    K::F64 => {
                        let values = counter
                            .map(|_| Ok(reader.read_f64::<LittleEndian>()?))
                            .collect::<Result<Vec<_>>>()?;
                        F::F64(values)
                    }
                };

                Ok(field)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(fields))
    }

    fn read_line<R: BufRead>(reader: &mut R, field_defs: &Schema) -> Result<Self> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let tokens = line.split_ascii_whitespace().collect::<Vec<_>>();

        {
            let expect = field_defs.iter().map(|def| def.count as usize).sum();
            let error = Error::new_text_token_mismatch_error(expect, tokens.len());
            if tokens.len() != expect {
                return Err(error.into());
            }
        }

        let mut tokens_iter = tokens.into_iter();
        let fields = field_defs
            .iter()
            .map(|def| {
                let FieldDef { kind, count, .. } = *def;

                let counter = 0..count;

                let field = match kind {
                    ValueKind::I8 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::I8(values)
                    }
                    ValueKind::I16 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::I16(values)
                    }
                    ValueKind::I32 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::I32(values)
                    }
                    ValueKind::U8 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::U8(values)
                    }
                    ValueKind::U16 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::U16(values)
                    }
                    ValueKind::U32 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::U32(values)
                    }
                    ValueKind::F32 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::F32(values)
                    }
                    ValueKind::F64 => {
                        let values = counter
                            .map(|_| Ok(tokens_iter.next().unwrap().parse()?))
                            .collect::<Result<Vec<_>>>()?;
                        Field::F64(values)
                    }
                };

                Ok(field)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self(fields))
    }

    #[cfg(feature = "binary_compressed")]
    fn read_compressed_chunk<R: BufRead>(reader: &mut R, pcd_meta: &PcdMeta) -> Result<Vec<Field>> {
        let mut u32_bytes = [0_u8; 4];
        reader.read_exact(&mut u32_bytes)?;
        let compressed_len = u32::from_le_bytes(u32_bytes) as usize;
        reader.read_exact(&mut u32_bytes)?;
        let decompressed_len = u32::from_le_bytes(u32_bytes) as usize;

        if compressed_len == 0 || decompressed_len == 0 {
            return Ok(Vec::new());
        }

        let mut reader = {
            let mut compressed_data = vec![0_u8; compressed_len];
            reader.read_exact(&mut compressed_data)?;

            let mut decompressed_data = vec![0_u8; decompressed_len];
            unsafe {
                let real_decompressed_len = lzf_sys::lzf_decompress(
                    compressed_data.as_ptr() as *const std::ffi::c_void,
                    compressed_len as std::ffi::c_uint,
                    decompressed_data.as_mut_ptr() as *mut std::ffi::c_void,
                    decompressed_len as std::ffi::c_uint,
                );
                if real_decompressed_len != decompressed_len as u32 {
                    bail!(
                        "Failed in lzf_decompression, length mismatch: {real_decompressed_len} != {decompressed_len}"
                    );
                }
            }

            std::io::BufReader::new(std::io::Cursor::new(decompressed_data))
        };

        use Field as F;
        use ValueKind as K;

        let mut fields = Vec::with_capacity(pcd_meta.field_defs.len());

        for FieldDef { kind, count, .. } in pcd_meta.field_defs.iter() {
            let mut field = match kind {
                K::U8 => F::U8(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
                K::U16 => F::U16(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
                K::U32 => F::U32(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
                K::I8 => F::I8(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
                K::I16 => F::I16(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
                K::I32 => F::I32(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
                K::F32 => F::F32(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
                K::F64 => F::F64(Vec::with_capacity((pcd_meta.num_points * *count) as usize)),
            };

            for _ in 0..pcd_meta.num_points {
                match &mut field {
                    F::I8(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_i8()?);
                        }
                    }
                    F::I16(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_i16::<LittleEndian>()?);
                        }
                    }
                    F::I32(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_i32::<LittleEndian>()?);
                        }
                    }
                    F::U8(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_u8()?);
                        }
                    }
                    F::U16(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_u16::<LittleEndian>()?);
                        }
                    }
                    F::U32(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_u32::<LittleEndian>()?);
                        }
                    }
                    F::F32(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_f32::<LittleEndian>()?);
                        }
                    }
                    F::F64(field_data) => {
                        for _ in 0..*count {
                            field_data.push(reader.read_f64::<LittleEndian>()?);
                        }
                    }
                }
            }

            fields.push(field);
        }

        Ok(fields)
    }

    #[cfg(feature = "binary_compressed")]
    fn read_decompressed_chunk(
        fields_data: &[Field],
        index: usize,
        field_defs: &Schema,
    ) -> Option<Self> {
        if fields_data.is_empty() {
            return None;
        }

        let mut record_fields = Vec::with_capacity(field_defs.len());

        for (field_def, field) in field_defs.iter().zip(fields_data.iter()) {
            use Field as F;

            let start_off = index * field_def.count as usize;
            let end_off = start_off + field_def.count as usize;
            let counter = start_off..end_off;

            let row_filed = match field {
                F::I8(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::I8(values)
                }
                F::I16(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::I16(values)
                }
                F::I32(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::I32(values)
                }
                F::U8(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::U8(values)
                }
                F::U16(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::U16(values)
                }
                F::U32(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::U32(values)
                }
                F::F32(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::F32(values)
                }
                F::F64(field_values) => {
                    if field_values.len() < end_off {
                        return None;
                    }
                    let values = counter.map(|i| field_values[i]).collect::<Vec<_>>();
                    F::F64(values)
                }
            };

            record_fields.push(row_filed);
        }

        Some(Self(record_fields))
    }
}

// impl for primitive types

impl PcdDeserialize for u8 {
    fn is_dynamic() -> bool {
        false
    }

    fn read_spec() -> Vec<(Option<String>, ValueKind, Option<usize>)> {
        vec![(None, ValueKind::U8, Some(1))]
    }

    fn read_chunk<R: BufRead>(reader: &mut R, _field_defs: &Schema) -> Result<Self> {
        let value = reader.read_u8()?;
        Ok(value)
    }

    fn read_line<R: BufRead>(reader: &mut R, _field_defs: &Schema) -> Result<Self> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        Ok(line.parse()?)
    }

    #[cfg(feature = "binary_compressed")]
    fn read_compressed_chunk<R: BufRead>(
        _reader: &mut R,
        _pcd_meta: &PcdMeta,
    ) -> Result<Vec<Field>> {
        bail!("Only support binary_compressed data")
    }

    #[cfg(feature = "binary_compressed")]
    fn read_decompressed_chunk(
        _fields_data: &[Field],
        _index: usize,
        _field_defs: &Schema,
    ) -> Option<Self> {
        // Only support binary_compressed data
        None
    }
}

impl PcdDeserialize for i8 {
    fn is_dynamic() -> bool {
        false
    }

    fn read_spec() -> Vec<(Option<String>, ValueKind, Option<usize>)> {
        vec![(None, ValueKind::I8, Some(1))]
    }

    fn read_chunk<R: BufRead>(reader: &mut R, _field_defs: &Schema) -> Result<Self> {
        let value = reader.read_i8()?;
        Ok(value)
    }

    fn read_line<R: BufRead>(reader: &mut R, _field_defs: &Schema) -> Result<Self> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        Ok(line.parse()?)
    }

    #[cfg(feature = "binary_compressed")]
    fn read_compressed_chunk<R: BufRead>(
        _reader: &mut R,
        _pcd_meta: &PcdMeta,
    ) -> Result<Vec<Field>> {
        bail!("Only support binary_compressed data")
    }

    #[cfg(feature = "binary_compressed")]
    fn read_decompressed_chunk(
        _fields_data: &[Field],
        _index: usize,
        _field_defs: &Schema,
    ) -> Option<Self> {
        // Only support binary_compressed data
        None
    }
}

macro_rules! impl_primitive {
    ($ty:ty, $kind:ident, $read:ident) => {
        impl PcdDeserialize for $ty {
            fn is_dynamic() -> bool {
                false
            }

            fn read_spec() -> Vec<(Option<String>, ValueKind, Option<usize>)> {
                vec![(None, ValueKind::$kind, Some(1))]
            }

            fn read_chunk<R: BufRead>(reader: &mut R, _field_defs: &Schema) -> Result<Self> {
                let value = reader.$read::<LittleEndian>()?;
                Ok(value)
            }

            fn read_line<R: BufRead>(reader: &mut R, _field_defs: &Schema) -> Result<Self> {
                let mut line = String::new();
                reader.read_line(&mut line)?;
                Ok(line.parse()?)
            }

            #[cfg(feature = "binary_compressed")]
            fn read_compressed_chunk<R: BufRead>(
                _reader: &mut R,
                _pcd_meta: &PcdMeta,
            ) -> Result<Vec<Field>> {
                bail!("Only support binary_compressed data")
            }

            #[cfg(feature = "binary_compressed")]
            fn read_decompressed_chunk(
                _fields_data: &[Field],
                _index: usize,
                _field_defs: &Schema,
            ) -> Option<Self> {
                // Only support binary_compressed data
                None
            }
        }
    };
}

impl_primitive!(u16, U16, read_u16);
impl_primitive!(u32, U32, read_u32);
impl_primitive!(i16, I16, read_i16);
impl_primitive!(i32, I32, read_i32);
impl_primitive!(f32, F32, read_f32);
impl_primitive!(f64, F64, read_f64);
