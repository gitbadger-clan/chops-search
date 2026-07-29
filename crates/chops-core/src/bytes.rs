//! Minimal little-endian reader/writer for the binary artifacts.
//!
//! Hand-rolled on purpose: serde + a format crate would dominate the wasm
//! binary size for what is, in total, four fixed layouts.

use crate::FormatError;

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        if self.remaining() < n {
            return Err(FormatError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, FormatError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> Result<u32, FormatError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn f32(&mut self) -> Result<f32, FormatError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// u16 length-prefixed UTF-8 string.
    pub fn str16(&mut self) -> Result<&'a str, FormatError> {
        let n = self.u16()? as usize;
        let b = self.take(n)?;
        core::str::from_utf8(b).map_err(|_| FormatError::BadUtf8)
    }

    /// Raw i8 slice of length n (reinterpreted from the byte buffer).
    pub fn i8s(&mut self, n: usize) -> Result<Vec<i8>, FormatError> {
        let b = self.take(n)?;
        Ok(b.iter().map(|&x| x as i8).collect())
    }

    pub fn f32s(&mut self, n: usize) -> Result<Vec<f32>, FormatError> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.f32()?);
        }
        Ok(out)
    }
}

#[derive(Default)]
pub struct Writer {
    pub buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer { buf: Vec::new() }
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// u16 length-prefixed UTF-8 string. Panics if the string exceeds
    /// u16::MAX bytes — enforce upstream (titles/urls/tokens are short).
    pub fn str16(&mut self, s: &str) {
        assert!(s.len() <= u16::MAX as usize, "str16 field too long");
        self.u16(s.len() as u16);
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub fn i8s(&mut self, v: &[i8]) {
        self.buf.extend(v.iter().map(|&x| x as u8));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_scalars() {
        let mut w = Writer::new();
        w.u8(7);
        w.u16(65535);
        w.u32(123_456_789);
        w.f32(-0.125);
        w.str16("café");
        w.i8s(&[-128, 0, 127]);

        let mut r = Reader::new(&w.buf);
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u16().unwrap(), 65535);
        assert_eq!(r.u32().unwrap(), 123_456_789);
        assert_eq!(r.f32().unwrap(), -0.125);
        assert_eq!(r.str16().unwrap(), "café");
        assert_eq!(r.i8s(3).unwrap(), vec![-128, 0, 127]);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn truncated_errors() {
        let mut r = Reader::new(&[1, 2]);
        assert_eq!(r.u32(), Err(FormatError::Truncated));
    }
}
