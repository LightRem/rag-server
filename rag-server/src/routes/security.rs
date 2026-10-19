use crate::error::ApiError;
use bytes::Bytes;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

const MAX_REDIRECT_HOPS: usize = 5;

pub fn is_allowed_port(port: u16) -> bool {
    matches!(port, 80 | 443 | 8080 | 8443)
}

pub fn is_forbidden_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            let octets = ipv4.octets();
            ipv4.is_loopback()
                || ipv4.is_unspecified()
                || (octets[0] == 0)
                || (octets[0] == 10)
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 169 && octets[1] == 254)
                || (octets[0] == 172 && (16..=31).contains(&octets[1]))
                || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                || (octets[0] == 192 && octets[0] == 0 && octets[2] == 2)
                || (octets[0] == 192 && octets[1] == 168)
                || (octets[0] == 198 && (18..=19).contains(&octets[1]))
                || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
                || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
                || (octets[0] >= 224)
        }
        IpAddr::V6(ipv6) => {
            let segs = ipv6.segments();
            if segs[0] == 0
                && segs[1] == 0
                && segs[2] == 0
                && segs[3] == 0
                && segs[4] == 0
                && (segs[5] == 0 || segs[5] == 0xffff)
            {
                let mapped_ipv4 = std::net::Ipv4Addr::new(
                    (segs[6] >> 8) as u8,
                    (segs[6] & 0xff) as u8,
                    (segs[7] >> 8) as u8,
                    (segs[7] & 0xff) as u8,
                );
                return is_forbidden_ip(IpAddr::V4(mapped_ipv4));
            }

            // NAT64 Well-Known Prefix (64:ff9b::/96)
            if segs[0] == 0x0040
                && segs[1] == 0xff9b
                && segs[2] == 0
                && segs[3] == 0
                && segs[4] == 0
            {
                let mapped_ipv4 = std::net::Ipv4Addr::new(
                    (segs[6] >> 8) as u8,
                    (segs[6] & 0xff) as u8,
                    (segs[7] >> 8) as u8,
                    (segs[7] & 0xff) as u8,
                );
                return is_forbidden_ip(IpAddr::V4(mapped_ipv4));
            }

            // 6to4 tunneling prefix (2002::/16)
            if segs[0] == 0x2002 {
                let mapped_ipv4 = std::net::Ipv4Addr::new(
                    (segs[1] >> 8) as u8,
                    (segs[1] & 0xff) as u8,
                    (segs[2] >> 8) as u8,
                    (segs[2] & 0xff) as u8,
                );
                return is_forbidden_ip(IpAddr::V4(mapped_ipv4));
            }

            ipv6.is_loopback()
                || ipv6.is_unspecified()
                || (segs[0] & 0xfe00) == 0xfc00
                || (segs[0] & 0xffc0) == 0xfe80
                || (segs[0] & 0xff00) == 0xff00
                || (segs[0] == 0x2001 && segs[1] == 0x0db8)
                || (segs[0] == 0x0100 && segs[1] == 0 && segs[2] == 0 && segs[3] == 0)
        }
    }
}

pub async fn resolve_and_validate_safe_url(
    url_str: &str,
) -> Result<(reqwest::Url, String, IpAddr, u16), ApiError> {
    let parsed = reqwest::Url::parse(url_str)
        .map_err(|e| ApiError::ValidationError(format!("Invalid image URL '{url_str}': {e}")))?;

    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(ApiError::ValidationError(format!(
            "Forbidden URL scheme '{}': only http and https are allowed",
            parsed.scheme()
        )));
    }

    let host_str = parsed
        .host_str()
        .ok_or_else(|| ApiError::ValidationError("Missing host in image URL".into()))?
        .to_string();

    let port = parsed.port_or_known_default().unwrap_or(80);
    if !is_allowed_port(port) {
        return Err(ApiError::ValidationError(format!(
            "Port {port} is forbidden for media fetching (allowed standard web ports: 80, 443, 8080, 8443)"
        )));
    }

    let addrs = tokio::net::lookup_host(format!("{host_str}:{port}"))
        .await
        .map_err(|e| {
            ApiError::ValidationError(format!("DNS resolution failed for '{host_str}': {e}"))
        })?;

    let mut safe_ip = None;
    for addr in addrs {
        if is_forbidden_ip(addr.ip()) {
            return Err(ApiError::ValidationError(format!(
                "Resolved IP '{}' for host '{}' is forbidden (SSRF protection)",
                addr.ip(),
                host_str
            )));
        }
        if safe_ip.is_none() {
            safe_ip = Some(addr.ip());
        }
    }

    let ip = safe_ip.ok_or_else(|| {
        ApiError::ValidationError(format!("No IP addresses resolved for '{host_str}'"))
    })?;

    Ok((parsed, host_str, ip, port))
}

pub async fn fetch_safe_remote_image(
    url: &str,
    max_allowed_bytes: usize,
) -> Result<Bytes, ApiError> {
    let mut current_url = url.to_string();
    let mut hops = 0;
    let mut cached_client: Option<(String, SocketAddr, reqwest::Client)> = None;

    let (resp, content_length) = loop {
        let (parsed_url, host_str, safe_ip, port) =
            resolve_and_validate_safe_url(&current_url).await?;
        let target_addr = SocketAddr::new(safe_ip, port);

        let client = match &cached_client {
            Some((h, addr, c)) if h == &host_str && addr == &target_addr => c.clone(),
            _ => {
                let new_c = reqwest::Client::builder()
                    .resolve(&host_str, target_addr)
                    .redirect(reqwest::redirect::Policy::none())
                    .timeout(Duration::from_secs(15))
                    .build()
                    .map_err(|e| {
                        ApiError::Internal(format!("Failed to build secure HTTP client: {e}"))
                    })?;
                cached_client = Some((host_str.clone(), target_addr, new_c.clone()));
                new_c
            }
        };

        let response =
            client.get(parsed_url.clone()).send().await.map_err(|e| {
                ApiError::ValidationError(format!("Failed to fetch image URL: {e}"))
            })?;

        if response.status().is_redirection() {
            hops += 1;
            if hops > MAX_REDIRECT_HOPS {
                return Err(ApiError::ValidationError(
                    "Too many redirects (max 5)".into(),
                ));
            }
            let loc_header = response
                .headers()
                .get(reqwest::header::LOCATION)
                .ok_or_else(|| {
                    ApiError::ValidationError("Redirect response missing Location header".into())
                })?;
            let loc_str = loc_header.to_str().map_err(|_| {
                ApiError::ValidationError("Invalid Location header encoding".into())
            })?;
            let next_url = parsed_url.join(loc_str).map_err(|e| {
                ApiError::ValidationError(format!("Invalid redirect target '{loc_str}': {e}"))
            })?;
            current_url = next_url.to_string();
            continue;
        }

        if !response.status().is_success() {
            return Err(ApiError::ValidationError(format!(
                "Image URL returned HTTP {}",
                response.status()
            )));
        }

        let cl = response.content_length();
        break (response, cl);
    };

    if let Some(cl) = content_length.filter(|&cl| cl > max_allowed_bytes as u64) {
        return Err(ApiError::PayloadTooLarge {
            size: cl as usize,
            max_allowed: max_allowed_bytes,
        });
    }

    let mut buffer = Vec::new();
    let mut stream = resp;
    while let Some(chunk) = stream
        .chunk()
        .await
        .map_err(|e| ApiError::ValidationError(format!("Failed to read image stream: {e}")))?
    {
        if buffer.len() + chunk.len() > max_allowed_bytes {
            return Err(ApiError::PayloadTooLarge {
                size: buffer.len() + chunk.len(),
                max_allowed: max_allowed_bytes,
            });
        }
        buffer.extend_from_slice(&chunk);
    }

    Ok(Bytes::from(buffer))
}
