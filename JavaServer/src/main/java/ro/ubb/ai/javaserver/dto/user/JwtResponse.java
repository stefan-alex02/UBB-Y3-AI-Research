package ro.ubb.ai.javaserver.dto.user;

import ro.ubb.ai.javaserver.enums.Role;
import lombok.Data;
import lombok.RequiredArgsConstructor;

@Data
@RequiredArgsConstructor
public class JwtResponse {
    private final String token;
    private String type = "Bearer";
    private final Long id;
    private final String username;
    private final Role role;
}
