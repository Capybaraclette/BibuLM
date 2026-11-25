use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use sha2::{Digest, Sha256};

const DPAPI_PREFIX: &str = "dpapi:v1:";

/// Computes lowercase hex SHA-256 digest of a UTF-8 string.
pub fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

/// Constant-time byte-slice comparison to avoid timing side-channels on token checks.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Masks a secret API token for safe storage/display (e.g. `rs-1234••••••abcd`).
pub fn mask_api_token(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.contains("••••") {
        return trimmed.to_string();
    }
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() <= 10 {
        return "••••••".to_string();
    }
    let prefix: String = chars.iter().take(7).collect();
    let suffix: String = chars[chars.len() - 4..].iter().collect();
    format!("{prefix}••••••{suffix}")
}

/// Encrypts a secret string at rest using Windows DPAPI (`CryptProtectData`), scoped to
/// the current Windows user account. On non-Windows platforms returns the value untouched
/// (database file permissions are restricted to `0600`).
pub fn encrypt_secret(plain: &str) -> String {
    if plain.is_empty() || plain.starts_with(DPAPI_PREFIX) {
        return plain.to_string();
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(cipher) = dpapi_protect(plain.as_bytes()) {
            return format!("{DPAPI_PREFIX}{}", BASE64.encode(cipher));
        }
    }
    plain.to_string()
}

/// Decrypts a secret stored by [`encrypt_secret`], transparently passing through legacy
/// plaintext values for backward compatibility.
pub fn decrypt_secret(stored: &str) -> String {
    if let Some(b64) = stored.strip_prefix(DPAPI_PREFIX) {
        #[cfg(target_os = "windows")]
        {
            if let Ok(cipher) = BASE64.decode(b64)
                && let Ok(plain) = dpapi_unprotect(&cipher)
                && let Ok(s) = String::from_utf8(plain)
            {
                return s;
            }
        }
        let _ = b64;
        return String::new();
    }
    stored.to_string()
}

#[cfg(target_os = "windows")]
fn dpapi_protect(plain: &[u8]) -> Result<Vec<u8>, ()> {
    use std::ffi::c_void;
    use windows::Win32::Foundation::{HLOCAL, LocalFree};
    use windows::Win32::Security::Cryptography::{
        CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptProtectData,
    };

    let in_blob = CRYPT_INTEGER_BLOB {
        cbData: u32::try_from(plain.len()).map_err(|_| ())?,
        pbData: plain.as_ptr() as *mut u8,
    };
    let mut out_blob = CRYPT_INTEGER_BLOB::default();
    unsafe {
        CryptProtectData(
            &in_blob,
            None,
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut out_blob,
        )
        .map_err(|_| ())?;
        let slice = std::slice::from_raw_parts(out_blob.pbData, out_blob.cbData as usize);
        let vec = slice.to_vec();
        let _ = LocalFree(HLOCAL(out_blob.pbData as *mut c_void));
        Ok(vec)
    }
}

#[cfg(target_os = "windows")]
fn dpapi_unprotect(cipher: &[u8]) -> Result<Vec<u8>, ()> {
    use std::ffi::c_void;
    use windows::Win32::Foundation::{HLOCAL, LocalFree};
    use windows::Win32::Security::Cryptography::{
        CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptUnprotectData,
    };

    let in_blob = CRYPT_INTEGER_BLOB {
        cbData: u32::try_from(cipher.len()).map_err(|_| ())?,
        pbData: cipher.as_ptr() as *mut u8,
    };
    let mut out_blob = CRYPT_INTEGER_BLOB::default();
    unsafe {
        CryptUnprotectData(
            &in_blob,
            None,
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut out_blob,
        )
        .map_err(|_| ())?;
        let slice = std::slice::from_raw_parts(out_blob.pbData, out_blob.cbData as usize);
        let vec = slice.to_vec();
        let _ = LocalFree(HLOCAL(out_blob.pbData as *mut c_void));
        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_and_constant_time_and_mask_work() {
        let h1 = sha256_hex("rs-abc1234567890");
        let h2 = sha256_hex("rs-abc1234567890");
        assert_eq!(h1.len(), 64);
        assert!(constant_time_eq(h1.as_bytes(), h2.as_bytes()));
        assert!(!constant_time_eq(h1.as_bytes(), b"different"));
        assert_eq!(mask_api_token("rs-1234567890abcdef"), "rs-1234••••••cdef");
    }

    #[test]
    fn secret_encryption_roundtrips_and_handles_legacy_plaintext() {
        let raw = "hf_super_secret_token_12345";
        let enc = encrypt_secret(raw);
        #[cfg(target_os = "windows")]
        {
            assert!(enc.starts_with(DPAPI_PREFIX));
            assert_ne!(enc, raw);
        }
        assert_eq!(decrypt_secret(&enc), raw);
        assert_eq!(decrypt_secret("legacy_plain_token"), "legacy_plain_token");
    }
}
