package ro.ubb.ai.javaserver.dto.user;

import lombok.Data;
import ro.ubb.ai.javaserver.enums.Role;

import java.time.OffsetDateTime;

@Data
public class UserDTO {
    private Long id;
    private String username;
    private String name;
    private Role role;
    private OffsetDateTime createdAt;
}
