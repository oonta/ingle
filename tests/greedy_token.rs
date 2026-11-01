use ingle::sample::greedy_token;

#[test]
fn a_caller_receives_the_winning_token_id() {
    assert_eq!(greedy_token(&[0.1, 0.4, 0.2]), Some(1));
}
