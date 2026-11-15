/// Text that might still grow into a stop string, and is not sent yet.
#[derive(Debug, Default)]
pub struct StopHold {
    pending: String,
}

impl StopHold {
    /// Append `delta`. Returns the text that is safe to send, and whether a stop matched.
    pub fn push(&mut self, delta: &str, stops: &[String]) -> (String, bool) {
        self.pending.push_str(delta);
        if let Some(index) = find_stop(&self.pending, stops) {
            let visible = self.pending[..index].to_string();
            self.pending.clear();
            return (visible, true);
        }
        let hold = held_suffix(&self.pending, stops);
        let visible_len = self.pending.len() - hold.len();
        let visible = self.pending[..visible_len].to_string();
        self.pending = hold;
        (visible, false)
    }

    /// Release text that was held only because it might have become a stop string.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn find_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
}

fn held_suffix(text: &str, stops: &[String]) -> String {
    let mut best = "";
    for (index, _) in text.char_indices() {
        let suffix = &text[index..];
        let held = stops
            .iter()
            .any(|stop| !stop.is_empty() && stop.starts_with(suffix));
        if held && suffix.len() > best.len() {
            best = suffix;
        }
    }
    best.to_string()
}

#[cfg(test)]
mod tests {
    use super::StopHold;

    #[test]
    fn a_partial_stop_is_held_and_a_match_is_dropped() {
        let mut hold = StopHold::default();
        let stops = vec!["xyz".to_string()];
        assert_eq!(hold.push("ab", &stops), ("ab".to_string(), false));
        assert_eq!(hold.push("xy", &stops), (String::new(), false));
        assert_eq!(hold.push("z", &stops), (String::new(), true));
    }

    #[test]
    fn a_near_miss_is_released_whole() {
        let mut hold = StopHold::default();
        let stops = vec!["xyz".to_string()];
        assert_eq!(hold.push("xy", &stops), (String::new(), false));
        assert_eq!(hold.push("q", &stops), ("xyq".to_string(), false));
    }
}
